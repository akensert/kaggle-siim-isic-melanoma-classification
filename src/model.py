import numpy as np
import tensorflow as tf
import tqdm
from sklearn import metrics
import os


def sigmoid_focal_cross_entropy_with_logits(
    labels, logits, alpha=0.25, gamma=2.0):

    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")

    logits = tf.convert_to_tensor(logits)
    labels = tf.convert_to_tensor(labels, dtype=logits.dtype)

    # Get the cross_entropy for each entry
    ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)

    # If logits are provided then convert the predictions into probabilities
    pred_prob = tf.math.sigmoid(logits)

    p_t = (labels * pred_prob) + ((1 - labels) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
        alpha_factor = labels * alpha + (1 - labels) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.math.reduce_sum(
        alpha_factor * modulating_factor * ce, axis=-1)


class NeuralNet(tf.keras.Model):

    def __init__(self, engine, input_shape, classifier, pretrained_weights):

        super(NeuralNet, self).__init__()

        self.engine = engine(
            include_top=False,
            input_shape=input_shape,
            weights=pretrained_weights)

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = classifier

    def call(self, inputs, **kwargs):
        x = self.engine(inputs)
        x = self.pool(x)
        return self.classifier(x)


class DistributedModel:

    def __init__(self,
                 keras_model,
                 batch_size=8,
                 finetuned_weights=None,
                 optimizer=None,
                 strategy=None,
                 mixed_precision=False,
                 label_smoothing=0.0,
                 tta=1,
                 focal_loss=True,
                 save_best=False):

        self.keras_model = keras_model
        self.global_batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.strategy = strategy
        self.label_smoothing = label_smoothing
        self.tta = tta
        self.focal_loss = focal_loss
        self.save_best = save_best

        self.auc_metric = tf.keras.metrics.AUC()
        self.loss_metric = tf.keras.metrics.Mean()

        if finetuned_weights:
            self.keras_model.load_weights(finetuned_weights)

        if self.optimizer and self.mixed_precision:
            self.optimizer = \
                tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, loss_scale='dynamic')

        if self.strategy:
            self.global_batch_size *= self.strategy.num_replicas_in_sync

        if not(os.path.isdir('output/weights')) and save_best:
            os.makedirs('output/weights')

    def _compute_loss(self, labels, logits):
        if self.focal_loss:
            per_example_loss = sigmoid_focal_cross_entropy_with_logits(
                labels=labels, logits=logits, alpha=0.8, gamma=2.0)
        else:
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits)
        return tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=self.global_batch_size)

    @tf.function
    def _distributed_train_step(self, dist_inputs):

        def train_step(inputs):

            images, labels = inputs

            if self.label_smoothing:
                labels = (
                    labels * (1 - self.label_smoothing)
                    + 0.5 * self.label_smoothing
                )

            with tf.GradientTape() as tape:
                logits = self.keras_model(images, training=True)
                loss = self._compute_loss(labels, logits)
                self.loss_metric.update_state(loss)
                self.auc_metric.update_state(
                    tf.math.round(labels), tf.math.sigmoid(logits))
                if self.mixed_precision:
                    scaled_loss = self.optimizer.get_scaled_loss(loss)

            if self.mixed_precision:
                scaled_gradients = tape.gradient(
                    scaled_loss, self.keras_model.trainable_variables)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(loss, self.keras_model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.keras_model.trainable_variables))

            return loss

        per_replica_loss = self.strategy.run(train_step, args=(dist_inputs,))
        return self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    @tf.function
    def _distributed_predict_step(self, dist_inputs):

        def predict_step(inputs):
            images, labels = inputs
            logits = self.keras_model(images, training=False)
            return tf.math.sigmoid(logits), labels

        preds, trues = self.strategy.run(predict_step, args=(dist_inputs,))
        if tf.is_tensor(preds):
            return [preds], [trues]
        else:
            return preds.values, trues.values

    def fit(self, ds):

        ds = self.strategy.experimental_distribute_dataset(ds)
        ds = tqdm.tqdm(ds)

        for i, inputs in enumerate(ds):
            loss = self._distributed_train_step(inputs)
            epoch_loss = self.loss_metric.result().numpy()
            epoch_auc = self.auc_metric.result().numpy()
            ds.set_description(
                "Scores [{:.4f}, {:.4f}] : Loss [{:.4f}, {:.4f}]".format(
                    self.epochs_score,
                    self.last_epoch_score,
                    self.loss_metric.result().numpy(),
                    self.auc_metric.result().numpy()
                )
            )

        self.loss_metric.reset_states()
        self.auc_metric.reset_states()

    def predict(self, ds):

        ds = self.strategy.experimental_distribute_dataset(ds.repeat(self.tta))
        ds = tqdm.tqdm(ds)

        preds_accum = np.zeros([0, 1], dtype=np.float32)
        trues_accum = np.zeros([0, 1], dtype=np.float32)
        for inputs in ds:
            preds, trues = self._distributed_predict_step(inputs)

            for pred, true in zip(preds, trues):
                preds_accum = np.concatenate([preds_accum, pred.numpy()], axis=0)
                trues_accum = np.concatenate([trues_accum, true.numpy()], axis=0)

        preds_accum = preds_accum.reshape((self.tta, -1)).mean(axis=0)
        trues_accum = trues_accum.reshape((self.tta, -1)).mean(axis=0).round()

        return preds_accum, trues_accum

    def fit_and_predict(self, fold, epochs, train_ds, valid_ds, test_ds):

        valid_preds_accum = np.zeros([0,], dtype=np.float32)
        test_preds_accum = np.zeros([0,], dtype=np.float32)

        self.epochs_score = 0.
        self.last_epoch_score = 0.
        self.best_score = 0.
        for epoch in range(epochs):

            # fit for an epoch
            self.fit(train_ds)

            # predict on validation set and add to accumulator
            preds, trues = self.predict(valid_ds)
            valid_preds_accum = np.concatenate([valid_preds_accum, preds], axis=0)

            # compute auc score and save model if best_score
            self.last_epoch_score = metrics.roc_auc_score(
                trues, valid_preds_accum.reshape((epoch+1, -1))[-1])

            if self.last_epoch_score > self.best_score and self.save_best:
                self.best_score = self.last_epoch_score
                self.keras_model.save_weights(f'output/weights/model-{fold}-{epoch}.h5')

            self.epochs_score = metrics.roc_auc_score(
                trues, np.average(
                    valid_preds_accum.reshape((epoch+1, -1)),
                    axis=0, weights=[2**i for i in range(epoch+1)]))

            # predict on test set and add to accumulator
            preds, _ = self.predict(test_ds)
            test_preds_accum = np.concatenate([test_preds_accum, preds], axis=0)

        return valid_preds_accum, test_preds_accum
