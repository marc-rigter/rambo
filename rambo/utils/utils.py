import numpy as np

def normalize(obs, obs_mean=None, obs_std=None):
    if (obs_mean is None) or (obs_std is None):
        return obs
    else:
        return (obs - obs_mean) / obs_std

def unnormalize(obs_norm, obs_mean=None, obs_std=None):
    if (obs_mean is None) or (obs_std is None):
        return obs_norm
    else:
        return obs_norm * obs_std + obs_mean
