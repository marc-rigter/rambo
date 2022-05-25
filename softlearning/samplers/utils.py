from copy import deepcopy

import numpy as np

from softlearning import replay_pools
from . import (
    dummy_sampler,
    extra_policy_info_sampler,
    base_sampler,
    simple_sampler)


def get_sampler_from_variant(variant, *args, **kwargs):
    SAMPLERS = {
        'DummySampler': dummy_sampler.DummySampler,
        'ExtraPolicyInfoSampler': (
            extra_policy_info_sampler.ExtraPolicyInfoSampler),
        'Sampler': base_sampler.BaseSampler,
        'SimpleSampler': simple_sampler.SimpleSampler,
    }

    sampler_params = variant['sampler_params']
    sampler_type = sampler_params['type']

    sampler_args = deepcopy(sampler_params.get('args', ()))
    sampler_kwargs = deepcopy(sampler_params.get('kwargs', {}))

    sampler = SAMPLERS[sampler_type](
        *sampler_args, *args, **sampler_kwargs, **kwargs)

    return sampler


def rollout(env,
            policy,
            path_length,
            obs_mean=None,
            obs_std=None,
            callback=None,
            render_mode=None,
            break_on_terminal=True):
    """
    If the observations from the true environment need to be rescaled
    before being passed to the policy, obs_mean and obs_std should be
    provided as arguments. If they are none the observations will not be
    rescaled.
    """
    observation_space = env.observation_space
    action_space = env.action_space

    pool = replay_pools.SimpleReplayPool(
        observation_space, action_space, max_size=path_length)
    sampler = simple_sampler.SimpleSampler(
        max_path_length=path_length,
        min_pool_size=None,
        batch_size=None
    )

    sampler.initialize(env, policy, pool, obs_mean, obs_std)

    images = []
    infos = []

    t = 0
    for t in range(path_length):
        observation, reward, terminal, info = sampler.sample()
        infos.append(info)

        if callback is not None:
            callback(observation)

        if render_mode is not None:
            if render_mode == 'rgb_array':
                image = env.render(mode=render_mode)
                # import pdb; pdb.set_trace()
                # image = env._env.sim.render(mode='offscreen')
                images.append(image)
            else:
                env.render()

        if terminal:
            policy.reset()
            if break_on_terminal: break

    if hasattr(env._env, "target_goal") and hasattr(env._env, "get_xy"):
        to_goal = np.asarray(env._env.target_goal - env._env.get_xy())
        dist = np.linalg.norm(to_goal)
        infos.append({"end_dist": dist, "vec_to_goal": to_goal})
    assert pool._size == t + 1

    path = pool.batch_by_indices(
        np.arange(pool._size),
        observation_keys=getattr(env, 'observation_keys', None))
    path['infos'] = infos

    if render_mode == 'rgb_array':
        path['images'] = np.stack(images, axis=0)

    return path


def rollouts(n_paths, *args, **kwargs):
    paths = [rollout(*args, **kwargs) for i in range(n_paths)]
    return paths
