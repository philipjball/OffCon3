import copy

import numpy as np
import torch
import torch.nn.functional as F

from utils import ReplayPool
from networks import Policy, StochasticPolicy, DoubleQFunc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OffPolicyAgent:

    def __init__(self, seed, state_dim, action_dim,
                 action_lim=1, lr=3e-4, gamma=0.99,
                 tau=5e-3, batch_size=256, hidden_size=256,
                 update_interval=2, buffer_size=1e6):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.action_lim = action_lim

        torch.manual_seed(seed)

        # aka critic
        self.q_funcs = DoubleQFunc(state_dim, action_dim, hidden_size=hidden_size).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size).to(device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.replay_pool = ReplayPool(capacity=int(buffer_size))

        self._update_counter = 0

    def reallocate_replay_pool(self, new_size: int):
        assert new_size != self.replay_pool.capacity, "Error, you've tried to allocate a new pool which has the same length"
        new_replay_pool = ReplayPool(capacity=new_size)
        new_replay_pool.initialise(self.replay_pool)
        self.replay_pool = new_replay_pool

    @property
    def is_soft(self):
        raise NotImplementedError

    def get_action(self, state, state_filter=None, deterministic=False):
        raise NotImplementedError

    def update_target(self):
        raise NotImplementedError

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        raise NotImplementedError

    def update_policy(self, state_batch):
        raise NotImplementedError

    def optimize(self, n_updates, state_filter=None):
        q1_loss, q2_loss, pi_loss, a_loss = 0, 0, None, None
        for i in range(n_updates):
            samples = self.replay_pool.sample(self.batch_size)
            if state_filter:
                state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
            else:
                state_batch = torch.FloatTensor(samples.state).to(device)
                nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)
            action_batch = torch.FloatTensor(samples.action).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.real_done).to(device).unsqueeze(1)
            
            # update q-funcs
            q1_loss_step, q2_loss_step = self.update_q_functions(state_batch, action_batch, reward_batch, nextstate_batch, done_batch)
            q_loss_step = q1_loss_step + q2_loss_step
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()
            
            self._update_counter += 1

            q1_loss += q1_loss_step.detach().item()
            q2_loss += q2_loss_step.detach().item()

            if self._update_counter % self.update_interval == 0:
                if not pi_loss:
                    pi_loss = 0
                # update policy
                for p in self.q_funcs.parameters():
                    p.requires_grad = False
                pi_loss_step = self.update_policy(state_batch)
                # if there's a soft policy (i.e., max-ent), then we need to update target entropy
                if self.is_soft:
                    if not a_loss:
                        a_loss = 0
                    pi_loss_step, a_loss_step = pi_loss_step
                    self.temp_optimizer.zero_grad()
                    a_loss_step.backward()
                    self.temp_optimizer.step()
                    a_loss += a_loss_step.detach().item()
                self.policy_optimizer.zero_grad()
                pi_loss_step.backward()
                self.policy_optimizer.step()
                for p in self.q_funcs.parameters():
                    p.requires_grad = True
                # update target policy and q-functions using Polyak averaging
                self.update_target()
                pi_loss += pi_loss_step.detach().item()

        return q1_loss, q2_loss, pi_loss, a_loss


class TD3_Agent(OffPolicyAgent):

    def __init__(self, seed, state_dim, action_dim,
                 action_lim=1, lr=3e-4, gamma=0.99,
                 tau=5e-3, batch_size=256, hidden_size=256,
                 update_interval=2, buffer_size=1e6,
                 target_noise=0.2, target_noise_clip=0.5, explore_noise=0.1):

        super().__init__(seed, state_dim, action_dim, action_lim,
                         lr, gamma, tau,
                         batch_size, hidden_size,
                         update_interval, buffer_size)

        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.explore_noise = explore_noise

        # TD3 also has a target POLICY
        self.target_policy = copy.deepcopy(self.policy)
        for p in self.target_policy.parameters():
            p.requires_grad = False

    def get_action(self, state, state_filter=None, deterministic=False):
        if state_filter:
            state = state_filter(state)
        state = torch.Tensor(state).view(1,-1).to(device)
        with torch.no_grad():
            action = self.policy(state)
        if not deterministic:
            action += self.explore_noise * torch.randn_like(action)
        action.clamp_(-self.action_lim, self.action_lim)
        return np.atleast_1d(action.squeeze().cpu().numpy())
    
    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
            for target_pi_param, pi_param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_pi_param.data.copy_(self.tau * pi_param.data + (1.0 - self.tau) * target_pi_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        with torch.no_grad():
            nextaction_batch = self.target_policy(nextstate_batch)
            target_noise = self.target_noise * torch.randn_like(nextaction_batch)
            target_noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
            nextaction_batch += target_noise
            nextaction_batch.clamp_(-self.action_lim, self.action_lim)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def update_policy(self, state_batch):
        action_batch = self.policy(state_batch)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (-qval_batch).mean()
        return policy_loss

    @property
    def is_soft(self):
        return False


class SAC_Agent(OffPolicyAgent):

    def __init__(self, seed, state_dim, action_dim,
                 action_lim=1, lr=3e-4, gamma=0.99,
                 tau=5e-3, batch_size=256, hidden_size=256,
                 update_interval=1, buffer_size=1e6,
                 target_entropy=None):
        
        super().__init__(seed, state_dim, action_dim, action_lim,
                         lr, gamma, tau,
                         batch_size, hidden_size,
                         update_interval, buffer_size)

        self.target_entropy = target_entropy if target_entropy else -action_dim

        # SAC uses the reparameterisation trick to propagate gradients through the variance head for entropy control
        self.policy = StochasticPolicy(state_dim, action_dim, hidden_size=hidden_size).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # aka temperature for target entropy
        self._log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.temp_optimizer = torch.optim.Adam([self._log_alpha], lr=lr)
    
    def get_action(self, state, state_filter=None, deterministic=False):
        if state_filter:
            state = state_filter(state)
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(device))
        if deterministic:
            return mean.squeeze().cpu().numpy()
        return np.atleast_1d(action.squeeze().cpu().numpy())

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * (q_target - self.alpha * logprobs_batch)
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def update_policy(self, state_batch):
        """Also updates the temperature parameter"""
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    @property
    def is_soft(self):
        return True

    @property
    def alpha(self):
        return self._log_alpha.exp()


class TDS_Agent(OffPolicyAgent):

    def __init__(self, seed, state_dim, action_dim,
                 action_lim=1, lr=3e-4, gamma=0.99,
                 tau=5e-3, batch_size=256, hidden_size=256,
                 update_interval=2, buffer_size=1e6,
                 target_noise=0.2, target_noise_clip=0.5, explore_noise=0.1):

        super().__init__(seed, state_dim, action_dim, action_lim,
                         lr, gamma, tau,
                         batch_size, hidden_size,
                         update_interval, buffer_size)

        self.policy = StochasticPolicy(state_dim, action_dim, hidden_size=hidden_size).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.explore_noise = explore_noise

    def get_action(self, state, state_filter=None, deterministic=False):
        if state_filter:
            state = state_filter(state)
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(device))
        if deterministic:
            return mean.squeeze().cpu().numpy()
        return np.atleast_1d(action.squeeze().cpu().numpy())

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        with torch.no_grad():
            nextaction_batch, _, nextmean_batch = self.policy(nextstate_batch, get_logprob=False)
            target_noise = nextaction_batch - nextmean_batch
            nextaction_batch = nextmean_batch + target_noise
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def update_policy(self, state_batch):
        action_batch, _, _ = self.policy(state_batch, get_logprob=False)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (-qval_batch).mean()
        return policy_loss

    @property
    def is_soft(self):
        return False
