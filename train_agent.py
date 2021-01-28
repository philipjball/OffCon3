import random
from argparse import ArgumentParser
from collections import deque

import gym
from gym.wrappers import RescaleAction
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from agents import TD3_Agent, SAC_Agent, TDS_Agent, MEPG_Agent
from utils import MeanStdevFilter, Transition, make_gif, make_checkpoint


def train_agent_model_free(agent, env, params):
    
    update_timestep = params['update_every_n_steps']
    seed = params['seed']
    log_interval = 1000
    checkpoint_interval = params['checkpoint_interval']
    n_random_actions = params['n_random_actions']
    n_evals = params['n_evals']
    n_collect_steps = params['n_collect_steps']
    use_statefilter = params['obs_filter']
    save_model = params['save_model']
    make_gif = params['make_gif']
    total_timesteps = params['total_timesteps']

    assert n_collect_steps > agent.batch_size, "We must initially collect as many steps as the batch size!"

    avg_length = 0
    time_step = 0
    cumulative_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    episode_rewards = []
    episode_steps = []

    if use_statefilter:
        state_filter = MeanStdevFilter(env.env.observation_space.shape[0])
    else:
        state_filter = None

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    max_steps = env.spec.max_episode_steps

    writer = SummaryWriter(log_dir=params['experiment_name'])

    if params["load_model_path"]:
        samples_number = agent.load_checkpoint(params["load_model_path"], params['env'])

    while samples_number < total_timesteps:
        time_step = 0
        episode_reward = 0
        i_episode += 1
        log_episode += 1
        state = env.reset()
        if state_filter:
            state_filter.update(state)
        done = False

        while (not done):
            cumulative_log_timestep += 1
            cumulative_timestep += 1
            time_step += 1
            samples_number += 1
            if samples_number < n_random_actions:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, state_filter=state_filter)
            nextstate, reward, done, _ = env.step(action)
            # if we hit the time-limit, it's not a 'real' done; we don't want to assign low value to those states
            real_done = False if time_step == max_steps else done
            agent.replay_pool.push(Transition(state, action, reward, nextstate, real_done))
            state = nextstate
            if state_filter:
                state_filter.update(state)
            episode_reward += reward
            # update if it's time
            if cumulative_timestep % update_timestep == 0 and cumulative_timestep > n_collect_steps:
                q1_loss, q2_loss, pi_loss, a_loss = agent.optimize(update_timestep, state_filter=state_filter)
                n_updates += 1
            # logging
            if cumulative_timestep % log_interval == 0 and cumulative_timestep > n_collect_steps:
                writer.add_scalar('Loss/Q-func_1', q1_loss, n_updates)
                writer.add_scalar('Loss/Q-func_2', q2_loss, n_updates)
                #TODO: This may not work; fix this
                if pi_loss:
                    writer.add_scalar('Loss/policy', pi_loss, n_updates)
                if a_loss:
                    writer.add_scalar('Loss/target_entropy', a_loss, n_updates)
                avg_length = np.mean(episode_steps)
                running_reward = np.mean(episode_rewards)
                eval_reward, action_var = evaluate_agent(env, agent, state_filter, n_starts=n_evals)
                writer.add_scalar('Reward/Train', running_reward, cumulative_timestep)
                writer.add_scalar('Reward/Test', eval_reward, cumulative_timestep)
                writer.add_scalar('Reward/Action_Var', action_var, cumulative_timestep)
                print('Episode {} \t Samples {} \t Avg length: {} \t Test reward: {} \t Train reward: {} \t Action Variance: {} \t Number of Updates: {}'.format(i_episode, samples_number, avg_length, eval_reward, running_reward, action_var, n_updates))
                episode_steps = []
                episode_rewards = []
            if cumulative_timestep % checkpoint_interval == 0:
                if make_gif:
                    make_gif(agent, env, cumulative_timestep, state_filter)
                if save_model:
                    make_checkpoint(agent, cumulative_timestep, params['env'], params['save_replay_pool'])

        episode_steps.append(time_step)
        episode_rewards.append(episode_reward)


def evaluate_agent(env, agent, state_filter, n_starts=1):
    reward_sum = 0
    all_actual_actions = []
    all_random_actions = []
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            action = agent.get_action(state, state_filter=state_filter, deterministic=True)
            action_rand = agent.get_action(state, state_filter=state_filter, deterministic=False)
            all_actual_actions.append(action)
            all_random_actions.append(action_rand)
            nextstate, reward, done, _ = env.step(action)
            reward_sum += reward
            state = nextstate
    all_actual_actions = np.array(all_actual_actions)
    all_random_actions = np.array(all_random_actions)
    action_var = ((all_random_actions - all_actual_actions)**2).mean(1).sum() / (all_actual_actions.shape[0] - 1)
    return reward_sum / n_starts, action_var


def get_agent_and_update_params(seed, state_dim, action_dim, params):

    params_old = params
    params = params_old.copy()

    if not params['yaml_config']:
        yaml_config = './configs/{}_config.yml'.format(params['alg'])

    with open(yaml_config, 'r') as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        # update overall args with the yaml file
        for config in yaml_config['args']:
            if config in params:
                # overwrite Nones, else leave it alone
                if not params[config]:
                    params[config] = yaml_config['args'][config]

    alg_config = yaml_config['alg_config']

    if params['alg'] == 'td3':
        agent = TD3_Agent(seed, state_dim, action_dim,
                          action_lim=alg_config['action_lim'], lr=alg_config['lr'], gamma=alg_config['gamma'],
                          tau=alg_config['tau'], batch_size=alg_config['batch_size'], hidden_size=alg_config['hidden_size'],
                          update_interval=alg_config['update_interval'], buffer_size=alg_config['buffer_size'],
                          target_noise=alg_config['target_noise'], target_noise_clip=alg_config['target_noise_clip'], explore_noise=alg_config['explore_noise'])
    elif params['alg'] == 'sac':
        agent = SAC_Agent(seed, state_dim, action_dim, 
                          action_lim=alg_config['action_lim'], lr=alg_config['lr'], gamma=alg_config['gamma'],
                          tau=alg_config['tau'], batch_size=alg_config['batch_size'], hidden_size=alg_config['hidden_size'],
                          update_interval=alg_config['update_interval'], buffer_size=alg_config['buffer_size'],
                          target_entropy=alg_config['target_entropy'])
    elif params['alg'] == 'mepg':
        agent = MEPG_Agent(seed, state_dim, action_dim, 
                          action_lim=alg_config['action_lim'], lr=alg_config['lr'], gamma=alg_config['gamma'],
                          tau=alg_config['tau'], batch_size=alg_config['batch_size'], hidden_size=alg_config['hidden_size'],
                          update_interval=alg_config['update_interval'], buffer_size=alg_config['buffer_size'],
                          target_entropy=alg_config['target_entropy'])
    elif params['alg'] == 'tds':
        agent = TDS_Agent(seed, state_dim, action_dim, 
                          action_lim=alg_config['action_lim'], lr=alg_config['lr'], gamma=alg_config['gamma'],
                          tau=alg_config['tau'], batch_size=alg_config['batch_size'], hidden_size=alg_config['hidden_size'],
                          update_interval=alg_config['update_interval'], buffer_size=alg_config['buffer_size'],
                          target_noise=alg_config['target_noise'], target_noise_clip=alg_config['target_noise_clip'], explore_noise=alg_config['explore_noise'])
    else:
        raise Exception('algorithm {} not supported'.format(params['alg']))

    print("Using {} policy optimizer...".format(params['alg'].upper()))

    return params, agent


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--alg', type=str, default='td3', choices={'td3', 'sac', 'tds', 'mepg'})
    parser.add_argument('--yaml_config', type=str, default=None)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    parser.add_argument('--update_every_n_steps', type=int, default=None)
    parser.add_argument('--n_random_actions', type=int, default=None)
    parser.add_argument('--n_collect_steps', type=int, default=None)
    parser.add_argument('--n_evals', type=int, default=1)
    parser.add_argument('--total_timesteps', type=int, default=1e7)
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--make_gif', dest='make_gif', action='store_true')
    parser.add_argument('--checkpoint_interval', type=int, default=500000)
    parser.add_argument('--save_replay_pool', type=bool, default=False)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.set_defaults(obs_filter=False)
    parser.set_defaults(save_model=False)

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']
    env = gym.make(params['env'])
    env = RescaleAction(env, -1, 1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    params, agent = get_agent_and_update_params(seed, state_dim, action_dim, params)

    train_agent_model_free(agent=agent, env=env, params=params)


if __name__ == '__main__':
    main()
