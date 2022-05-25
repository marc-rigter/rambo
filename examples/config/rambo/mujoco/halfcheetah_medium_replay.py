from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'halfcheetah',
    'task': 'medium-replay-v2',
    'exp_name': 'halfcheetah_medium_replay'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/halfcheetah-medium-replay-v2',
    'rollout_length': 5,
    'adversary_loss_weighting': 3e-4,
})
