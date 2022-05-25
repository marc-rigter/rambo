import os
import glob
import pickle
import gzip
import pdb
import numpy as np
def restore_pool(
        replay_pool,
        experiment_root,
        save_path=None,
        normalize_states=True,
        normalize_rewards=False
    ):
    if 'd4rl' in experiment_root:
        data = restore_pool_d4rl(replay_pool, experiment_root[5:])

    data, obs_mean, obs_std = normalise_data(data, normalize_states, normalize_rewards, experiment_root)
    replay_pool.add_samples(data)

    print('[ mbpo/off_policy ] Replay pool has size: {}'.format(replay_pool.size))
    return obs_mean, obs_std

def restore_pool_d4rl(replay_pool, name):
    import gym
    import d4rl
    data = d4rl.qlearning_dataset(gym.make(name))
    data['rewards'] = np.expand_dims(data['rewards'], axis=1)
    data['terminals'] = np.expand_dims(data['terminals'], axis=1)
    return data

def normalise_data(data, normalize_states, normalize_rewards, dataset_name):
    obs_mean = None
    obs_std = None
    if (not normalize_states) and (not normalize_rewards):
        return data, obs_mean, obs_std

    obs = data["observations"]
    next_obs = data["next_observations"]
    rewards = data["rewards"]

    # compute mean and std across subsample of data
    inds = np.floor(np.linspace(0, obs.shape[0]-1, num=10000)).astype(int)

    # subtract 1 from antmaze rewards per IQL paper
    if 'antmaze' in dataset_name:
        data['rewards'] -= 1

    if normalize_rewards:
        rew_std = np.std(rewards[inds, :], axis=0) + 1e-6
        data['rewards'] = rewards / rew_std

    if normalize_states:
        obs_mean = np.mean(obs[inds, :], axis=0)
        obs_std = np.std(obs[inds, :], axis=0) + 1e-6 # avoid division by zero
        obs_norm = (obs - obs_mean) / obs_std
        next_obs_norm = (next_obs - obs_mean) / obs_std

        data["observations"] = obs_norm
        data["next_observations"] = next_obs_norm

    return data, obs_mean, obs_std
