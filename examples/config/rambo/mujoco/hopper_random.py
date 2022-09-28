from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'hopper',
    'task': 'random-v2',
    'exp_name': 'hopper_random'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/hopper-random-v2',
    'rollout_length': 2,
    'adversary_loss_weighting': 3e-4,
})
