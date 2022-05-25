from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'walker2d',
    'task': 'medium-replay-v2',
    'exp_name': 'walker2d_medium_replay'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/walker2d-medium-replay-v2',
    'rollout_length': 5,
    'adversary_loss_weighting': 0.,
})
