from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_datasets as tfds


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, tf.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * tf.range(0, T + 1, dtype=tf.float32) / T + beta1
    sqrt_beta_t = tf.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = tf.math.log(alpha_t)
    alphabar_t = tf.math.exp(tf.cumsum(log_alpha_t, axis=0))

    sqrtab = tf.sqrt(alphabar_t)
    oneover_sqrta = 1 / tf.sqrt(alpha_t)

    sqrtmab = tf.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(tf.keras.layers.Layer):
    def __init__(
        self,
        eps_model: tf.keras.Model,
        betas: Tuple[float, float],
        n_T: int,
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            setattr(self, k, tf.Variable(v, trainable=False))

        self.n_T = n_T
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = tf.random.uniform(shape=[tf.shape(x)[0]], minval=1, maxval=self.n_T,
                                dtype=tf.int32)  # t ~ Uniform(0, n_T)
        eps = tf.random.normal(shape=tf.shape(x))  # eps ~ N(0, 1)
        gathered_sqrtab = tf.gather(self.sqrtab, _ts, axis=0)
        gathered_sqrtmab = tf.gather(self.sqrtmab, _ts, axis=0)
        x_t = (
                gathered_sqrtab[:, None, None, None] * x
                + gathered_sqrtmab[:, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        return self.mse(eps, self.eps_model(x_t, _ts / self.n_T)),

    def sample(self, n_sample: int, size, device) -> tf.Tensor:

        x_i = tf.random.normal([n_sample] + list(size))  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = tf.random.normal([n_sample] + list(size)) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i

