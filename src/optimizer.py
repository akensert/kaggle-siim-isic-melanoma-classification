import tensorflow as tf


def get_optimizer(steps_per_epoch, lr_max, lr_min,
                  decay_epochs, warmup_epochs, power=1):

    if decay_epochs > 0:
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=lr_max,
            decay_steps=steps_per_epoch*decay_epochs,
            end_learning_rate=lr_min,
            power=power,
        )
    else:
        learning_rate_fn = lr_max

    if warmup_epochs > 0:
        learning_rate_fn = WarmUp(
            lr_start = lr_min,
            lr_end = lr_max,
            lr_fn = learning_rate_fn,
            warmup_steps=steps_per_epoch*warmup_epochs,
            power=power,
        )

    return tf.keras.optimizers.Adam(learning_rate_fn)


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, lr_start, lr_end, lr_fn, warmup_steps, power=1):
        super().__init__()
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_fn = lr_fn
        self.warmup_steps = warmup_steps
        self.power = power

    def __call__(self, step):
        global_step_float = tf.cast(step, tf.float32)
        warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
        warmup_percent_done = global_step_float / warmup_steps_float
        warmup_learning_rate = tf.add(tf.multiply(
            self.lr_start-self.lr_end,
            tf.math.pow(1-warmup_percent_done, self.power)), self.lr_end)
        return tf.cond(
            global_step_float < warmup_steps_float,
            lambda: warmup_learning_rate,
            lambda: self.lr_fn(step),
        )

    def get_config(self):
        return {
            "lr_start": self.lr_start,
            "lr_end": self.lr_end,
            "lr_fn": self.lr_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
        }


class AdaBound(tf.keras.optimizers.Optimizer):
    """AdaBound optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        final_learning_rate: float >= 0. Final learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        gamma: float >= 0. Convergence speed of the bound function.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: Weight decay weight.
        amsbound: boolean. Whether to apply the AMSBound variant of this
            algorithm.
    # References
        - [Adaptive Gradient Methods with Dynamic Bound of Learning Rate]
          (https://openreview.net/forum?id=Bkg3g2R9FX)
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
    """
    def __init__(self,
                 learning_rate=0.001,
                 final_learning_rate=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 gamma=1e-3,
                 epsilon=None,
                 weight_decay=0.0,
                 amsbound=False,
                 name='AdaBound', **kwargs):
        super(AdaBound, self).__init__(name, **kwargs)

        self._set_hyper('learning_rate', kwargs.get('learning_rate', learning_rate))
        self._set_hyper('final_learning_rate', kwargs.get('final_learning_rate', final_learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('gamma', gamma)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsbound = amsbound
        self.weight_decay = weight_decay
        self.base_lr = learning_rate

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            self.add_slot(var, 'vhat')

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        vhat = self.get_slot(var, 'vhat')

        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)

        gamma = self._get_hyper('gamma')
        final_lr = self._get_hyper('final_learning_rate')

        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        base_lr_t = tf.convert_to_tensor(self.base_lr)
        t = tf.cast(self.iterations + 1, var_dtype)

        # Applies bounds on actual learning rate
        step_size = lr_t * (tf.math.sqrt(1. - tf.math.pow(beta_2_t, t)) /
                          (1. - tf.math.pow(beta_1_t, t)))

        final_lr = final_lr * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma * t + 1.))
        upper_bound = final_lr * (1. + 1. / (gamma * t))

        # apply weight decay
        if self.weight_decay != 0.:
            grad += self.weight_decay * var

        # Compute moments
        m_t = (beta_1_t * m) + (1. - beta_1_t) * grad
        v_t = (beta_2_t * v) + (1. - beta_2_t) * tf.math.square(grad)

        if self.amsbound:
            vhat_t = tf.math.maximum(vhat, v_t)
            denom = (tf.math.sqrt(vhat_t) + epsilon_t)
        else:
            vhat_t = vhat
            denom = (tf.math.sqrt(v_t) + self.epsilon)

        # Compute the bounds
        step_size_p = step_size * tf.ones_like(denom)
        step_size_p_bound = step_size_p / denom
        bounded_lr_t = m_t * tf.math.minimum(tf.math.maximum(step_size_p_bound,
                                             lower_bound), upper_bound)

        # Setup updates
        m_t = tf.compat.v1.assign(m, m_t)
        vhat_t = tf.compat.v1.assign(vhat, vhat_t)

        with tf.control_dependencies([m_t, v_t, vhat_t]):
            p_t = var - bounded_lr_t
            param_update = tf.compat.v1.assign(var, p_t)

            return tf.group(*[param_update, m_t, v_t, vhat_t])

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Sparse data is not supported yet")

    def get_config(self):
        config = super(AdaBound, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'final_learning_rate': self._serialize_hyperparameter('final_learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'gamma': self._serialize_hyperparameter('gamma'),
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'amsbound': self.amsbound,
        })
        return config
