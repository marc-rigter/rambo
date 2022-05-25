import numpy as np
import tensorflow as tf

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        done = np.array([False]).repeat(len(obs))
        done = done[:,None]
        return done

    @staticmethod
    def termination_fn_tf(obs, act, next_obs):
        return obs[:, 0] * 0.

    @staticmethod
    def recompute_reward_fn(obs, act, next_obs, rew):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        new_rew = -(rew + 0.1 * np.sum(np.square(act))) - 0.1 * np.sum(np.square(act))
        return new_rew
