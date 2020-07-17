import tensorflow as tf

def sigmoid_focal_cross_entropy_with_logits(labels,
                                            logits,
                                            alpha=0.25,
                                            gamma=2.0):

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
