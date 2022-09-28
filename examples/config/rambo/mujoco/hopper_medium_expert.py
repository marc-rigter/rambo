from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'hopper',
    'task': 'medium-expert-v2',
    'exp_name': 'hopper_medium_expert'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/hopper-medium-expert-v2',
    'rollout_length': 5,
    'adversary_loss_weighting': 3e-4,
})
