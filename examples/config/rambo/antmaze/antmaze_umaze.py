from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'antmaze',
    'task': 'umaze-v0',
    'exp_name': 'antmaze_umaze'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/antmaze-umaze-v0',
    'rollout_length': 5,
    'adversary_loss_weighting': 3e-4,
    'rollout_random': False,
    'pretrain_bc': False
})
