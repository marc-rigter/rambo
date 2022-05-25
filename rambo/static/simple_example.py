import numpy as np
import tensorflow as tf

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        t = next_obs[:, 1]
        done = (t > 2.5)
        done = done[:,None]
        return done

    @staticmethod
    def termination_fn_tf(obs, act, next_obs):
        t = next_obs[:, 1]
        done = tf.cast(tf.greater(t, 2.5), tf.float32)
        return done
