from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'halfcheetah',
    'task': 'medium-v2',
    'exp_name': 'halfcheetah_medium'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/halfcheetah-medium-v2',
    'rollout_length': 5,
    'adversary_loss_weighting': 3e-4,
})
