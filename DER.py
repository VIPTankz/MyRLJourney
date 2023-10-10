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
import kornia.augmentation as aug
import kornia
import pickle
from ChurnData import ChurnData
import matplotlib.pyplot as plt
#from torchsummary import summary
from Identify import Identify
import mgzip
import math

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.1):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir,atoms,Vmax,Vmin, device):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.atoms = atoms
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.DELTA_Z = (self.Vmax - self.Vmin) / (self.atoms - 1)
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(4, 32, 5, stride=5, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=5, padding=0)

        self.fc1V = NoisyLinear(64 * 3 * 3, 256)
        self.fc1A = NoisyLinear(64 * 3 * 3, 256)
        self.fcV2 = NoisyLinear(256, atoms)
        self.fcA2 = NoisyLinear(256, n_actions * atoms)

        self.register_buffer("supports", T.arange(Vmin, Vmax + self.DELTA_Z, self.DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=0.00015)
        self.loss = nn.MSELoss()
        self.device = device
        self.use_noise = True
        print("Device: " + str(self.device),flush=True)
        self.to(self.device)

    def conv(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x
    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

    def fc_val(self, x):
        x = F.relu(self.fc1V(x))
        x = self.fcV2(x)

        return x

    def fc_adv(self, x):
        x = F.relu(self.fc1A(x))
        x = self.fcA2(x)

        return x

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1, self.atoms)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, self.atoms)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, self.atoms)).view(t.size())

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


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
        self.N_ATOMS = 51

        self.memory = ReplayMemory(max_mem_size, self.n, self.gamma, device)

        self.net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims, name='DER_eval',
                                          chkpt_dir=self.chkpt_dir,atoms=self.N_ATOMS,Vmax=self.Vmax,Vmin=self.Vmin, device=device)

        self.tgt_net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,name='DER_next',
                                          chkpt_dir=self.chkpt_dir,atoms=self.N_ATOMS,Vmax=self.Vmax,Vmin=self.Vmin, device=device)

        self.net.train()
        self.tgt_net.train()

        for param in self.tgt_net.parameters():
            param.requires_grad = False

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

    def get_grad_steps(self):
        return self.grad_steps

    def set_eval_mode(self):
        self.net.eval()
        self.tgt_net.eval()
        self.eval_mode = True

    def choose_action(self, observation):
        if np.random.random() > self.epsilon or not self.eval_mode:
            with T.no_grad():
                self.net.reset_noise()
                state = T.tensor(np.array([observation]), dtype=T.float).to(self.net.device)

                advantage = self.net.qvals(state)
                x = T.argmax(advantage).item()
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

        idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)

        states = states.clone().detach().to(self.net.device)
        rewards = rewards.clone().detach().to(self.net.device)
        dones = dones.clone().detach().to(self.net.device).squeeze()
        actions = actions.clone().detach().to(self.net.device)
        states_ = next_states.clone().detach().to(self.net.device)

        #use this code to check your states are correct
        """
        plt.imshow(states[0][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        plt.show()
        """

        distr_v, qvals_v = self.net.both(states)
        state_action_values = distr_v[range(self.batch_size), actions.data]
        state_log_sm_v = F.log_softmax(state_action_values, dim=1)

        with torch.no_grad():
            self.tgt_net.reset_noise()
            next_distr_v, next_qvals_v = self.tgt_net.both(states_)
            action_distr_v, action_qvals_v = self.net.both(states_)

            next_actions_v = action_qvals_v.max(1)[1]

            next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
            next_best_distr_v = self.tgt_net.apply_softmax(next_best_distr_v)
            next_best_distr = next_best_distr_v.data.cpu()

            proj_distr = distr_projection(next_best_distr, rewards.cpu(), dones.cpu(), self.Vmin, self.Vmax, self.N_ATOMS,
                                          self.gamma ** self.n)

            proj_distr_v = proj_distr.to(self.net.device)

        loss_v = -state_log_sm_v * proj_distr_v
        weights = T.squeeze(weights)
        loss_v = weights.to(self.net.device) * loss_v.sum(dim=1)

        loss = loss_v.mean()

        loss.backward()
        T.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.net.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            print("Completed " + str(self.grad_steps) + " gradient steps")

        self.memory.update_priorities(idxs, loss_v.cpu().detach().numpy())

def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = T.zeros((batch_size, n_atoms), dtype=T.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).type(T.int64)
        u = np.ceil(b_j).type(T.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).type(T.int64)
        u = np.ceil(b_j).type(T.int64)
        eq_mask = u == l
        eq_dones = T.clone(dones)
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = T.clone(dones)
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr