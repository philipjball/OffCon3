# **Off** Policy RL for **Con**tinuous **Con**trol **Con**solidated (OffCon<sup>3</sup>)
A minimal PyTorch implementation from scratch of the two model-free state of the art off-policy continuous control algoirthms:

* Twin Delayed DDPG (TD3)
* Soft Actor Critic (SAC)

This repo consolidates, where possible, the code between these two similar off-policy methods, and highlights the similarities (i.e., optimisation scheme) and differences (i.e., stochastic v.s. deterministic policies).

Heavily based on my other repos, [TD3-PyTorch](https://github.com/fiorenza2/TD3-PyTorch) and [SAC-PyTorch](https://github.com/fiorenza2/SAC-PyTorch). If you only want to use one of these algorithms, those repos may serve you better.

## Implementation Details

### TD3
This code borrows the hyperparameters from [Scott Fujimoto's implementation](https://github.com/sfujim/TD3) but with one difference, which is that the network architecture is the same as the SAC paper (barring the additional output units for log-variance). This means there's an extra layer of 256 hidden units.

### SAC
This code implements the follow up paper [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905), which includes a learned target entropy hyperparameter. 

## Instructions

### Quick Start 

Simply run:

`python train_agent.py`

for default args. Changeable args are:
```
--env: String of environment name (Default: HalfCheetah-v2)
--alg: String of policy optimizer (Default: td3; Choices: {td3, sac})
--yaml_config: String of YAML config file for either TD3 or SAC (Default: None)
--seed: Int of seed (Default: 100)
--use_obs_filter: Boolean that is true when used (seems to degrade performance)
--update_every_n_steps: Int of how many env steps we take before optimizing the agent (Default: 1, ratio of steps v.s. backprop is tied to 1:1)
--n_random_actions: Int of how many random steps we take to 'seed' the replay pool (Default: 25000 for TD3, 10000 for SAC)
--n_collect_steps: Int of how steps we collect before training  (Default: 1000)
--n_evals: Int of how many episodes we run an evaluation for (Default: 1)
--save_model: Boolean that is true when used (saves model when GIFs are made, loading and running is left as an exercise for the reader (or until I get around to it))
```

### Details
There are algorithm specific YAML files stored in `./configs/` for TD3 and SAC. These contain default configurations and hyperparameters that work well in OpenAI MuJoCo tasks. If no file is specified in the `--yaml_config` argument, then 

## Results

