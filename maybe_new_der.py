import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memory import ReplayMemory
import numpy as np
from collections import deque
from networks import C51_small
from Identify import Identify
import mgzip
import math


class Agent():
    def __init__(self, n_actions, input_dims, device,
                 max_mem_size=100000, total_frames=100000, lr=0.0001,
                 game=None, run=None, name=None):

        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims

        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.min_sampling_size = 1600

        self.chkpt_dir = ""

        self.run = run
        self.algo_name = name
        self.game = game

        #MAKE SURE YOU CHECKED THE NAME AND DATA COLLETION

        # IMPORTANT params, check these
        self.n = 20
        self.gamma = 0.99
        self.batch_size = 32
        self.replace_target_cnt = 2000
        self.replay_ratio = 1
        self.network = "normal"

        #data collection
        self.collecting_churn_data = False
        self.action_gap_data = False
        self.reward_proportions = False
        self.gen_data = False
        self.identify_data = False

        if self.identify_data:
            self.identify = Identify(self.min_sampling_size)

        if self.gen_data:
            self.batch_size = 1

        #c51
        self.Vmax = 10
        self.Vmin = -10
        self.atoms = 51

        self.memory = ReplayMemory(max_mem_size, self.n, self.gamma, device)

        self.net = C51_small(self.n_actions, self.atoms, device)
        self.tgt_net = C51_small(self.n_actions, self.atoms, device)

        for param in self.tgt_net.parameters():
            param.requires_grad = False

        #n-step
        self.n = 20

        if self.gen_data:
            self.net.optimizer = optim.RMSprop(self.net.parameters(), lr=lr)
            self.tgt_net.optimizer = optim.RMSprop(self.net.parameters(), lr=lr)

        self.env_steps = 0
        self.grad_steps = 0
        self.reset_churn = False
        self.second_save = False

        self.epsilon = 0.001 #used for eval only

        self.start_churn = self.min_sampling_size
        self.churn_sample = 256
        self.churn_dur = 23000
        self.second_churn = 75000
        self.total_churn = 0
        self.churn_data = []
        self.churn_actions = np.array([0 for i in range(self.n_actions)], dtype=np.float64)
        self.total_actions = np.array([0 for i in range(self.n_actions)], dtype=np.int64)
        self.count_since_reset = 0
        self.game = game

        self.action_swaps = np.zeros((self.n_actions, self.n_actions), dtype=np.int64)

        self.action_changes = np.zeros(self.n_actions, dtype=np.int64)
        self.action_changes2 = np.zeros(self.n_actions, dtype=np.int64)
        self.action_changes3 = np.zeros(self.n_actions, dtype=np.int64)
        self.steps = 1000
        self.lists = []
        self.lists2 = []
        self.lists3 = []

        self.total_change = 0
        self.total_gen_change = 0
        self.total_targetted_change = 0

        self.target_percent_list = []
        self.target_percent_mov_avg = 0.5

        self.reward_target_avg = []
        self.bootstrap_target_avg = []

        self.action_gaps = []

        self.replay_ratio_cnt = 0
        self.eval_mode = False

        self.priority_weight_increase = (1 - 0.4) / (total_frames - self.min_sampling_size)

        self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device=device)  # Support (range) of z
        self.delta_z = (self.Vmax - self.Vmin) / (self.atoms - 1)

    def get_grad_steps(self):
        return self.grad_steps

    def set_eval_mode(self):
        self.net.val()
        self.tgt_net.val()
        self.eval_mode = True

    def choose_action(self, state):
        if np.random.random() > self.epsilon or not self.eval_mode:
            with torch.no_grad():
                return (self.net(torch.from_numpy(state).unsqueeze(0)) * self.support).sum(2).argmax(1).item()
        else:
            x = np.random.choice(self.action_space)
        return x

    def store_transition(self, state, action, reward, state_, done):
        self.memory.append(torch.from_numpy(state), action, reward, done)
        self.env_steps += 1
        self.total_actions[action] += 1

    def replace_target_network(self):
        self.tgt_net.load_state_dict(self.net.state_dict())

    def save_models(self):
        self.net.save_checkpoint()
        self.tgt_net.save_checkpoint()

    def load_models(self):
        self.net.load_checkpoint()
        self.tgt_net.load_checkpoint()

    def learn(self):
        if self.replay_ratio < 1:
            if self.replay_ratio_cnt == 0:
                self.learn_call()
            self.replay_ratio_cnt = (self.replay_ratio_cnt + 1) % (int(1 / self.replay_ratio))
        else:
            for i in range(self.replay_ratio):
                self.learn_call()

    def learn_call(self):

        if self.env_steps < self.min_sampling_size:
            return

        self.memory.priority_weight = min(self.memory.priority_weight + self.priority_weight_increase, 1)

        self.net.optimizer.zero_grad()

        if self.grad_steps % self.replace_target_cnt == 0:
            self.replace_target_network()

        idxs, states, actions, returns, next_states, nonterminals, weights = self.memory.sample(self.batch_size)

        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.tgt_net.reset_noise()  # Sample new target net noise
            pns = self.tgt_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(
                self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.gamma ** self.n) * self.support.unsqueeze(
                0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.net.optimiser.step()

        self.memory.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

        self.grad_steps += 1