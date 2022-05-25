import numpy as np
import tensorflow as tf

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:,None]
        return done

    @staticmethod
    def termination_fn_tf(obs, act, next_obs):
        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        alive = tf.cast(tf.math.reduce_all(tf.math.is_finite(next_obs), 1), tf.float32) * \
                tf.cast(tf.math.reduce_all(tf.less(tf.math.abs(next_obs[:, 1:]), 100), 1), tf.float32) * \
                tf.cast(tf.greater(height, 0.7), tf.float32) * \
                tf.cast(tf.less(tf.math.abs(angle), 0.2), tf.float32)

        done = tf.ones_like(next_obs[:, 0]) - alive
        return done
