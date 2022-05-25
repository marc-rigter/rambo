import numpy as np
import tensorflow as tf

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done =  (height > 0.8) \
                    * (height < 2.0) \
                    * (angle > -1.0) \
                    * (angle < 1.0)
        done = ~not_done
        done = done[:,None]
        return done

    @staticmethod
    def termination_fn_tf(obs, act, next_obs):

        height = next_obs[:, 0]
        angle = next_obs[:, 1]

        alive = tf.cast(tf.greater(height, 0.8), tf.float32) * \
                tf.cast(tf.less(height, 2.0), tf.float32) * \
                tf.cast(tf.greater(angle, -1.0), tf.float32) * \
                tf.cast(tf.less(angle, 1), tf.float32)

        done = tf.ones_like(next_obs[:, 0]) - alive
        return done
