# Robust Adversarial Model-Based Offline Reinforcement Learning (RAMBO)

Official code to reproduce the experiments for [RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning](https://arxiv.org/abs/2204.12581). This implementation builds upon the code for [MOPO](https://github.com/tianheyu927/mopo).

For a PyTorch implementation of RAMBO, please see [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit).

## Installation
1. Install [MuJoCo 2.1.0](https://github.com/deepmind/mujoco/releases) to `~/.mujoco/mujoco210`.
2. Create a conda environment and install RAMBO.
```
cd rambo
conda create --name rambo python=3.7
conda activate rambo
pip install -e .
pip install -r requirements.txt
```

## Usage
Configuration files can be found in `examples/config/`. For example, to run the hopper-medium task from the D4RL benchmark, use the following.

```
rambo run_example examples.development --config examples.config.rambo.mujoco.hopper_medium --seed 0 --gpus 1
```


#### Logging

By default, TensorBoard logs are generated in the "logs" directory. The code is also set up to log using Weights and Biases (WandB). To enable the use of WandB, set "log_wandb" to True in the configuration file.


## Citing RAMBO

```
@article{rigter2022rambo,
  title={RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning},
  author={Rigter, Marc and Lacerda, Bruno and Hawes, Nick},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```
