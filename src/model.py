import tensorflow as tf
import os
import tqdm
from sklearn import metrics

from addons import sigmoid_focal_cross_entropy_with_logits

class NeuralNet(tf.keras.Model):

    def __init__(self, engine, input_shape, pretrained_weights):

        super(NeuralNet, self).__init__()

        self.engine = engine(
            include_top=False,
            input_shape=input_shape,
            weights=pretrained_weights)

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.concat = tf.keras.layers.Concatenate()

        self.sequential_meta = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        self.sequential_merged = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, dtype='float32')
        ])

    def call(self, inputs, **kwargs):

        if isinstance(inputs, dict):
            images = inputs['image']
            site = inputs['anatom_site_general_challenge']
            sex = inputs['sex']
            age = inputs['age_approx']
        else:
            # when model.build(input_shape) is called
            images = inputs[0]
            site = inputs[1]
            sex = inputs[2]
            age = inputs[3]

        x1 = self.engine(images)
        x1 = self.pool(x1)
        x2 = tf.concat([site, sex, age], axis=-1)
        x2 = self.sequential_meta(x2)
        x3 = self.concat([x1, x2])
        x3 = self.sequential_merged(x3)
        return x3


class DistributedModel:

    def __init__(self,
                 engine,
                 input_shape=(384, 384, 3),
                 pretrained_weights=None,
                 finetuned_weights=None,
                 batch_size=8,
                 optimizer=None,
                 strategy=None,
                 mixed_precision=False,
                 label_smoothing=0.0,
                 tta=1,
                 focal_loss=True,
                 save_best=None):

        self.keras_model = NeuralNet(
            engine=engine,
            input_shape=input_shape,
            pretrained_weights=pretrained_weights)
        self.keras_model.build(
            [[None, *input_shape], [None, 7], [None, 1], [None, 1]])
        if finetuned_weights:
            self.keras_model.load_weights(finetuned_weights)
        self._initial_weights = self.keras_model.get_weights()
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

        if self.optimizer and self.mixed_precision:
            self.optimizer = \
                tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, loss_scale='dynamic')

        if self.strategy:
            self.global_batch_size *= self.strategy.num_replicas_in_sync

        if save_best and not(os.path.isdir(save_best)):
            os.makedirs(save_best)

    def reset_weights(self):
        self.keras_model.set_weights(self._initial_weights)

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

            if self.label_smoothing:
                target = (
                    inputs['target'] * (1 - self.label_smoothing)
                    + 0.5 * self.label_smoothing
                )
            else:
                target = inputs['target']

            with tf.GradientTape() as tape:

                logits = self.keras_model(inputs, training=True)
                loss = self._compute_loss(target, logits)
                self.loss_metric.update_state(loss)
                self.auc_metric.update_state(
                    tf.math.round(target), tf.math.sigmoid(logits))
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
            logits = self.keras_model(inputs, training=False)
            return tf.math.sigmoid(logits), inputs['image_name'], inputs['target']

        preds, image_names, trues = self.strategy.run(predict_step, args=(dist_inputs,))
        if tf.is_tensor(preds):
            return [preds], [image_names], [trues]
        else:
            return preds.values, image_names.values, trues.values

    def fit(self, ds):

        ds = self.strategy.experimental_distribute_dataset(ds)
        ds = tqdm.tqdm(ds)

        for i, inputs in enumerate(ds):
            loss = self._distributed_train_step(inputs)
            epoch_loss = self.loss_metric.result().numpy()
            epoch_auc = self.auc_metric.result().numpy()
            ds.set_description(
                "valid AUC {:.4f} : Loss/AUC [{:.4f}, {:.4f}]".format(
                    self.auc_score,
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
        names_accum = np.zeros([0, 1], dtype=str)
        trues_accum = np.zeros([0, 1], dtype=np.float32)

        for inputs in ds:
            preds, names, trues = self._distributed_predict_step(inputs)

            for pred, name, true in zip(preds, names, trues):
                preds_accum = np.concatenate([preds_accum, pred.numpy()], axis=0)
                names_accum = np.concatenate([names_accum, name.numpy()], axis=0)
                trues_accum = np.concatenate([trues_accum, true.numpy()], axis=0)

        preds_accum = preds_accum.reshape((self.tta, -1)).mean(axis=0)
        names_accum = names_accum.reshape((self.tta, -1))[0]
        trues_accum = trues_accum.reshape((self.tta, -1)).mean(axis=0).round()

        return preds_accum, names_accum, trues_accum

    def fit_and_predict(self, fold, epochs, train_ds, valid_ds, test_ds):

        self.auc_score = 0.
        self.best_score = 0.
        for epoch in range(epochs):

            # fit for an epoch
            self.fit(train_ds)

            # predict on validation set
            valid_preds, valid_names, valid_trues = self.predict(valid_ds)

            # compute auc score and save model if best_score
            self.auc_score = metrics.roc_auc_score(valid_trues, valid_preds)

            if self.auc_score > self.best_score:
                self.best_score = self.auc_score
                best_valid_preds = valid_preds.copy()
                if self.save_best:
                    self.keras_model.save_weights(
                        self.save_best+f'{self.keras_model.layers[0].name}-{fold}-{epoch}.h5')
                # predict on test set
                test_preds, test_names, _ = self.predict(test_ds)

        return valid_preds, valid_names, test_preds, test_names
