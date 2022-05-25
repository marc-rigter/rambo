from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'halfcheetah',
    'task': 'random-v2',
    'exp_name': 'halfcheetah_random'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/halfcheetah-random-v2',
    'rollout_length': 5,
    'adversary_loss_weighting': 0,
})
