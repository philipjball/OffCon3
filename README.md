# **Off** Policy RL for **Con**tinuous **Con**trol **Con**solidated (OffCon<sup>3</sup>)
Code for the OffCon<sup>3</sup> paper available [here](https://arxiv.org/abs/2101.11331).

A minimal PyTorch implementation from scratch of the two model-free state of the art off-policy continuous control algoirthms:

* Twin Delayed DDPG (TD3)
* Soft Actor Critic (SAC)

This repo consolidates, where possible, the code between these two similar off-policy methods, and highlights the similarities (i.e., optimisation scheme) and differences (i.e., stochastic v.s. deterministic policies).

Heavily based on my other repos, [TD3-PyTorch](https://github.com/fiorenza2/TD3-PyTorch) and [SAC-PyTorch](https://github.com/fiorenza2/SAC-PyTorch). If you only want to use one of these algorithms, those repos may serve you better.

To cite this repo, please use the following BiBTex:
```
@misc{ball2021offcon3,
      title={OffCon$^3$: What is state of the art anyway?}, 
      author={Philip J. Ball and Stephen J. Roberts},
      year={2021},
      eprint={2101.11331},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Implementation Details

### TD3
This code implements the [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) paper, using SAC hyperparameters where appropriate (i.e., learning rate, network architecture).

### SAC
This code implements the follow up paper [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905), which includes a learned entropy trade-off hyperparameter. 

### TDS
As mentioned in the paper, this is SVG(0) with double-Q (or SAC without entropy); analysis shows this is essentially DDPG when trained on standard Gym MuJoCo.

## Instructions

### Quick Start 

Simply run:

`python train_agent.py`

for default args. Changeable args are:
```
--env: String of environment name (Default: HalfCheetah-v2)
--alg: String of policy optimizer (Default: td3; Choices: {td3, sac, tds})
--yaml_config: String of YAML config file for either TD3, SAC or TDS (Default: None)
--seed: Int of seed (Default: 100)
--use_obs_filter: Boolean that is true when used (seems to degrade performance, Default: False)
--update_every_n_steps: Int of how many env steps we take before optimizing the agent (Default: 1, ratio of steps v.s. backprop is tied to 1:1)
--n_random_actions: Int of how many random steps we take to 'seed' the replay pool (Default: 10000)
--n_collect_steps: Int of how steps we collect before training  (Default: 1000)
--n_evals: Int of how many episodes we run an evaluation for (Default: 1)
--checkpoint_interval: Int of how often to checkpoint model (i.e., saving, making gifs)
--save_model: Boolean that is true when used, saving the model parameters
--make_gif: Boolean that is true when used; makes
--save_replay_pool: Boolean that saves the replay pool along with the agent parameters (defaults to False, as this is very costly memory-wise)
--load_model_path: Path to the directory where model .pt files were saved; loads and resumes training from that snapshot
```

### Details
There are algorithm specific YAML files stored in `./configs/` for TD3 and SAC. These contain default configurations and hyperparameters that work well in OpenAI MuJoCo tasks. If no file is specified in the `--yaml_config` argument, then default YAMLs are loaded.

Also included is a `run_experiments.py` file, that allows the running of 5 simultaneous experiments with different seeds.

## Results

See [paper](https://arxiv.org/abs/2101.11331).

TL;DR: This seems to perform in the worst case, as well as author's code, and in the best case, significantly better.