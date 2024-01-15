import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ExperienceReplay import NStepExperienceReplay
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
from memory import ReplayMemory
import math
from EMA import EMA

class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = T.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.using_noise = True
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
    if self.using_noise:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class C51DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir,atoms,Vmax,Vmin, device, noisy, dueling, der_archit):
        super(C51DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.atoms = atoms
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.DELTA_Z = (self.Vmax - self.Vmin) / (self.atoms - 1)
        self.n_actions = n_actions
        self.dueling = dueling
        self.noisy = noisy
        self.der_archit = der_archit

        if not self.der_archit:
            self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3)
            output_size = 64 * 7 * 7
        else:
            self.conv1 = nn.Conv2d(4, 32, 5, stride=5, padding=0)
            self.conv2 = nn.Conv2d(32, 64, 5, stride=5, padding=0)
            output_size = 64 * 3 * 3

        if self.dueling:
            if self.noisy:
                self.fc1V = NoisyLinear(output_size, 256)
                self.fc1A = NoisyLinear(output_size, 256)
                self.fcV2 = NoisyLinear(256, atoms)
                self.fcA2 = NoisyLinear(256, n_actions * atoms)
            else:
                self.fc1V = nn.Linear(output_size, 256)
                self.fc1A = nn.Linear(output_size, 256)
                self.fcV2 = nn.Linear(256, atoms)
                self.fcA2 = nn.Linear(256, n_actions * atoms)
        else:
            if self.noisy:
                self.fc1 = NoisyLinear(64 * 7 * 7, 512)
                self.fc2 = NoisyLinear(512, n_actions * atoms)
            else:
                self.fc1 = nn.Linear(64 * 7 * 7, 512)
                self.fc2 = nn.Linear(512, n_actions * atoms)

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
        if not self.der_archit:
            x = F.relu(self.conv3(x))

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
        if self.dueling:
            val_out = self.fc_val(conv_out).view(batch_size, 1, self.atoms)
            adv_out = self.fc_adv(conv_out).view(batch_size, -1, self.atoms)
            adv_mean = adv_out.mean(dim=1, keepdim=True)
            return val_out + (adv_out - adv_mean)
        else:
            x = F.relu(self.fc1(conv_out))
            x = self.fc2(x).view(batch_size, -1, self.atoms)
            return x

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

    def disable_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.using_noise = False

    def enable_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.using_noise = True

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class QNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, device, noisy, batchnorm):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.n_actions = n_actions

        self.batchnorm = batchnorm

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm1d(512)

        self.noisy = noisy

        if self.noisy:

            self.fc1 = NoisyLinear(64 * 7 * 7, 512)
            self.fc2 = NoisyLinear(512, n_actions)
        else:
            self.fc1 = nn.Linear(64 * 7 * 7, 512)
            self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=0.00015)

        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def forward(self, observation):
        observation = T.div(observation, 255)
        observation = observation.view(-1, 4, 84, 84)
        observation = F.relu(self.conv1(observation))
        if self.batchnorm:
            observation = self.bn1(observation)
        observation = F.relu(self.conv2(observation))
        if self.batchnorm:
            observation = self.bn2(observation)
        observation = F.relu(self.conv3(observation))
        if self.batchnorm:
            observation = self.bn3(observation)
        observation = observation.view(-1, 64 * 7 * 7)
        observationQ = F.relu(self.fc1(observation))
        if self.batchnorm:
            observationQ = self.bn4(observationQ)
        Q = self.fc2(observationQ)

        return Q


    def reset_fcs(self):
        if self.noisy:
            self.fc1 = NoisyLinear(64 * 7 * 7, 512)
            self.fc2 = NoisyLinear(512, self.n_actions)
        else:
            self.fc1 = nn.Linear(64 * 7 * 7, 512)
            self.fc2 = nn.Linear(512, self.n_actions)

        self.to(self.device)



    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

    def disable_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.using_noise = False

    def enable_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.using_noise = True

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, device, noisy):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.n_actions = n_actions

        self.noisy = noisy

        if self.noisy:
            self.fc1V = NoisyLinear(64 * 7 * 7, 512)
            self.fc1A = NoisyLinear(64 * 7 * 7, 512)
            self.fc2V = NoisyLinear(512, 1)
            self.fc2A = NoisyLinear(512, n_actions)
        else:
            self.fc1V = nn.Linear(64 * 7 * 7, 512)
            self.fc1A = nn.Linear(64 * 7 * 7, 512)
            self.fc2V = nn.Linear(512, 1)
            self.fc2A = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=0.00015)

        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()


    def disable_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.using_noise = False

    def enable_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.using_noise = True

    def forward(self, observation):
        observation = T.div(observation, 255)
        observation = observation.view(-1, 4, 84, 84)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 64 * 7 * 7)
        observationV = F.relu(self.fc1V(observation))
        observationA = F.relu(self.fc1A(observation))
        V = self.fc2V(observationV)
        A = self.fc2A(observationA)

        Q = V + A - A.mean(dim=1, keepdim=True)

        return Q

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class EpsilonGreedy():
    def __init__(self):
        self.eps = 1.0
        self.steps = 5000
        self.eps_final = 0.1

    def update_eps(self):
        self.eps = max(self.eps - (self.eps - self.eps_final) / self.steps, self.eps_final)


class Agent():
    def __init__(self, n_actions, input_dims, device,
                 max_mem_size=100000, total_frames=100000, lr=0.0001,
                 game=None, run=None, name=None):

        self.epsilon = EpsilonGreedy()
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
        self.n = 10
        self.gamma = 0.967
        self.batch_size = 32
        self.duelling = False
        self.aug = False
        self.replace_target_cnt = 1
        self.replay_ratio = 1
        self.per = False
        self.annealing_n = True
        self.annealing_gamma = False
        self.target_ema = False
        self.double = True
        self.trust_regions = True  # Not implemented for c51
        self.trust_region_disable = False

        self.my_trust_region = False  # not implemented for c51
        self.my_trust_region2 = False

        self.resets = False  # not implemented for c51 or duelling

        self.noisy = False
        self.batch_norm = False  # only implemented for vanilla q networks

        self.c51 = False

        self.der_archit = False  # This is only implemented for c51

        if self.resets:
            self.reset_period = 40000
            self.sp_alpha = 0.8
            self.og_n = self.n
            self.og_gamma = self.gamma

        if self.c51:
            self.Vmax = 10
            self.Vmin = -10
            self.N_ATOMS = 51

        if self.noisy:
            self.epsilon.eps = 0.0
            self.epsilon.steps = 1
            self.epsilon.eps_final = 0.0

        if self.trust_regions:
            self.running_std = -999
            self.trust_alpha = 1
            self.replace_target_cnt = 1500

        if self.batch_norm:
            self.replace_target_cnt = 1

        self.trust_region_start = 5000

        if self.target_ema:
            self.ema_tau = 0.005

        if self.my_trust_region:
            self.target_ema = True
            self.ema_tau = 0.005
            self.trust_alpha = 0.3

        if self.my_trust_region2:
            self.running_std = -999
            self.network_queue_length = 30
            self.network_queue = deque([], maxlen=self.network_queue_length)
            self.target_queue_replace = 50
            self.std_tau = 0.001
            self.replace_target_cnt = 1500

        if self.annealing_gamma:
            self.final_gamma = 0.99
            self.anneal_steps_gamma = 10000
            self.gamma_inc = (self.final_gamma - self.gamma) / self.anneal_steps_gamma

        if self.annealing_n:
            self.final_n = 3
            self.anneal_steps = 30000
            self.n_dec = (self.n - self.final_n) / self.anneal_steps
            self.n_float = float(self.n)

        #data collection
        self.collecting_churn_data = True
        self.variance_data = False

        self.action_gap_data = False
        self.reward_proportions = False
        self.gen_data = False
        self.identify_data = False

        self.explosion = False

        if self.variance_data:
            self.variances = []

        if self.identify_data:
            self.identify = Identify(self.min_sampling_size)

        if self.gen_data:
            self.batch_size = 1

        if not self.per:
            self.memory = NStepExperienceReplay(input_dims, max_mem_size, self.batch_size, self.n, self.gamma)
        else:
            self.memory = ReplayMemory(max_mem_size, self.n, self.gamma, device)


        if self.duelling and not self.c51:
            self.net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                              input_dims=self.input_dims,
                                              name='lunar_lander_dueling_ddqn_q_eval',
                                              chkpt_dir=self.chkpt_dir, device=device, noisy=self.noisy)

            self.tgt_net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                              input_dims=self.input_dims,
                                              name='lunar_lander_dueling_ddqn_q_next',
                                              chkpt_dir=self.chkpt_dir, device=device, noisy=self.noisy)
        elif self.c51:
            self.net = C51DeepQNetwork(self.lr, self.n_actions,
                                           input_dims=self.input_dims, name='DER_eval',
                                           chkpt_dir=self.chkpt_dir, atoms=self.N_ATOMS, Vmax=self.Vmax, Vmin=self.Vmin,
                                           device=device, noisy=self.noisy, dueling=self.duelling, der_archit=self.der_archit)

            if not self.target_ema:
                self.tgt_net = C51DeepQNetwork(self.lr, self.n_actions,
                                                   input_dims=self.input_dims, name='DER_next',
                                                   chkpt_dir=self.chkpt_dir, atoms=self.N_ATOMS, Vmax=self.Vmax,
                                                   Vmin=self.Vmin, device=device, noisy=self.noisy, dueling=self.duelling, der_archit=self.der_archit)

                if self.replace_target_cnt > 1 or self.noisy:
                    self.churn_net = C51DeepQNetwork(self.lr, self.n_actions,
                                                   input_dims=self.input_dims, name='DER_next',
                                                   chkpt_dir=self.chkpt_dir, atoms=self.N_ATOMS, Vmax=self.Vmax,
                                                   Vmin=self.Vmin, device=device, noisy=self.noisy, dueling=self.duelling, der_archit=self.der_archit)

            if self.target_ema:
                self.tgt_net = EMA(self.net, self.ema_tau)
                self.tgt_net = C51DeepQNetwork(self.lr, self.n_actions,
                                                   input_dims=self.input_dims, name='DER_next',
                                                   chkpt_dir=self.chkpt_dir, atoms=self.N_ATOMS, Vmax=self.Vmax,
                                                   Vmin=self.Vmin, device=device, noisy=self.noisy, dueling=self.duelling)


        else:
            self.net = QNetwork(self.lr, self.n_actions,
                                           input_dims=self.input_dims, device=device, noisy=self.noisy, batchnorm=self.batch_norm)


            if not self.target_ema:
                self.tgt_net = QNetwork(self.lr, self.n_actions,
                                                   input_dims=self.input_dims, device=device, noisy=self.noisy, batchnorm=self.batch_norm)

                if self.replace_target_cnt > 1 or self.noisy:
                    self.churn_net = QNetwork(self.lr, self.n_actions,
                                            input_dims=self.input_dims, device=device, noisy=self.noisy, batchnorm=self.batch_norm)

            if self.target_ema:
                self.tgt_net = EMA(self.net, self.ema_tau)
                self.churn_net = QNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims, device=device, noisy=self.noisy, batchnorm=self.batch_norm)

        if self.c51:
            for param in self.tgt_net.parameters():
                param.requires_grad = False


        if self.gen_data:
            self.net.optimizer = optim.RMSprop(self.net.parameters(), lr=lr)
            self.tgt_net.optimizer = optim.RMSprop(self.net.parameters(), lr=lr)


        if self.trust_regions:
            self.replace_target_network()

        self.random_shift = nn.Sequential(nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
        self.intensity = Intensity(scale=0.1)

        self.env_steps = 0
        self.grad_steps = 0
        self.reset_churn = False
        self.second_save = False

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

        self.batch_q_vals = []

        if self.per:
            self.priority_weight_increase = (1 - 0.4) / (total_frames * self.replay_ratio - self.min_sampling_size)

    def shrink_perturb(self):
        #shink and perturb

        if self.duelling and not self.c51:
            new_net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                              input_dims=self.input_dims,
                                              name='lunar_lander_dueling_ddqn_q_next',
                                              chkpt_dir=self.chkpt_dir, device=self.net.device, noisy=self.noisy)
        elif self.c51:
            new_net = C51DeepQNetwork(self.lr, self.n_actions,
                                           input_dims=self.input_dims, name='DER_eval',
                                           chkpt_dir=self.chkpt_dir, atoms=self.N_ATOMS, Vmax=self.Vmax, Vmin=self.Vmin,
                                           device=self.net.device, noisy=self.noisy, dueling=self.duelling, der_archit=self.der_archit)

        else:
            new_net = QNetwork(self.lr, self.n_actions,
                                           input_dims=self.input_dims, device=self.net.device, noisy=self.noisy, batchnorm=self.batch_norm)

        with torch.no_grad():
            for param, target_param in zip(self.net.parameters(), new_net.parameters()):

                # it doesn't matter that this moves all params since fc layers got randomly initialised anyway
                param.data.copy_(self.sp_alpha * param.data +
                                        (1 - self.sp_alpha) * target_param.data)

    def get_grad_steps(self):
        return self.grad_steps

    def set_eval_mode(self):
        self.epsilon.eps_final = 0.05
        self.epsilon.eps = 0.05
        if self.noisy:
            self.net.disable_noise()

    def choose_action(self, observation):
        if self.identify_data and self.env_steps >= self.min_sampling_size:
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.net.device)

            q_vals = self.net.forward(state)
            action_exploit = T.argmax(q_vals).item()
            action_explore = np.random.choice(self.action_space)

            self.identify.Qvals.append(q_vals.detach().cpu().numpy())

            if np.random.random() > self.epsilon.eps:
                action = action_exploit
            else:
                action = action_explore

            return action

        if np.random.random() > self.epsilon.eps:
            with torch.no_grad():
                if self.noisy:
                    self.net.reset_noise()

            state = T.tensor(np.array([observation]), dtype=T.float).to(self.net.device)

            if self.batch_norm:
                self.net.eval()

            if not self.c51:
                if self.replay_ratio > 1:
                    q_vals = self.tgt_net.forward(state)
                else:
                    q_vals = self.net.forward(state)
            else:
                if self.replay_ratio > 1:
                    q_vals = self.tgt_net.qvals(state)
                else:
                    q_vals = self.net.qvals(state)
            action = T.argmax(q_vals).item()

            if self.batch_norm:
                self.net.train()

        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.env_steps += 1
        self.total_actions[action] += 1

        if self.per:
            self.memory.append(torch.from_numpy(state), action, reward, done)
        else:
            self.memory.store_transition(state, action, reward, state_, done)

        if self.identify_data:
            if self.memory.mem_cntr >= self.min_sampling_size:

                self.identify.states.append(state)
                self.identify.actions.append(action)
                self.identify.rewards.append(reward)
                self.identify.dones.append(done)

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

        self.net.optimizer.zero_grad()

        if (self.replace_target_cnt > 1 or self.target_ema or self.noisy) and  self.collecting_churn_data:
            self.churn_net.load_state_dict(self.net.state_dict())

        if self.annealing_n:
            self.n_float = max(self.final_n, self.n_float - self.n_dec)
            if self.n_float <= self.n - 1:
                self.n = int(round(self.n_float))
                self.memory.update_n(self.n)

        if self.annealing_gamma:
            self.gamma = min(self.final_gamma, self.gamma + self.gamma_inc)
            self.memory.discount = self.gamma

        if self.resets:
            if self.grad_steps % self.reset_period == self.reset_period - 1:
                self.net.reset_fcs()
                self.shrink_perturb()
                self.n = self.og_n
                self.n_float = self.og_n
                self.gamma = self.og_gamma

                self.memory.update_n(self.n)
                self.memory.discount = self.gamma

        if self.grad_steps % self.replace_target_cnt == 0 and not self.target_ema: # and not self.my_trust_region2
            self.replace_target_network()

        if self.target_ema:
            self.tgt_net.update()

        """if self.my_trust_region2:
            if self.grad_steps % self.target_queue_replace == 0:
                if len(self.network_queue) == self.network_queue_length:
                    self.tgt_net.load_state_dict(self.network_queue[0])

                self.network_queue.append(self.net.state_dict())"""

        with torch.no_grad():
            if self.noisy:
                self.tgt_net.reset_noise()

        if self.per:
            self.memory.priority_weight = min(self.memory.priority_weight + self.priority_weight_increase, 1)
            idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)
            if idxs is False:
                return
            states = states.clone().detach().to(self.net.device)
            rewards = rewards.clone().detach().to(self.net.device)
            dones = dones.clone().detach().to(self.net.device).squeeze()
            actions = actions.clone().detach().to(self.net.device)
            states_ = next_states.clone().detach().to(self.net.device)
        else:
            states, actions, rewards, new_states, dones = self.memory.sample_memory()
            states = T.tensor(states).to(self.net.device)
            rewards = T.tensor(rewards).to(self.net.device)
            dones = T.tensor(dones).to(self.net.device)
            actions = T.tensor(actions).to(self.net.device)
            states_ = T.tensor(new_states).to(self.net.device)

        indices = np.arange(self.batch_size)

        if self.aug:
            states = (self.intensity(self.random_shift(states.float()))).to(T.uint8)
            states_ = (self.intensity(self.random_shift(states_.float()))).to(T.uint8)
            states_policy_ = (self.intensity(self.random_shift(states_.float()))).to(T.uint8)
        else:
            states_policy_ = states_

        if not self.c51:
            if not self.batch_norm:
                q_pred = self.net.forward(states)  # states_aug

                if not self.trust_regions or not self.my_trust_region or not self.my_trust_region2:
                    q_targets = self.tgt_net.forward(states_)

                if self.double:
                    q_actions = self.net.forward(states_policy_)
                else:
                    q_actions = q_targets.clone().detach()

                if self.trust_regions or self.my_trust_region or self.my_trust_region2:
                    q_targets = q_actions.clone()
            else:
                states_and_next = torch.cat((states, states_))
                output = self.net.forward(states_and_next)
                q_pred, q_targets = torch.chunk(output, 2)
                q_actions = q_targets.clone().detach()

        else:
            distr_v, q_pred = self.net.both(states)
            state_action_values = distr_v[range(self.batch_size), actions.data]
            state_log_sm_v = F.log_softmax(state_action_values, dim=1)

            next_distr_v, next_qvals_v = self.tgt_net.both(states_)
            if self.double and not self.trust_regions:
                action_distr_v, action_qvals_v = self.net.both(states_)
            else:
                action_distr_v = next_distr_v
                action_qvals_v = next_qvals_v


        if self.identify_data or self.action_gap_data:
            q_pred_og = q_pred.detach().cpu()

        if not self.c51:
            q_pred = q_pred[indices, actions]

        if self.explosion:
            if self.grad_steps % 100 == 0:
                print("Grad Steps: " + str(self.grad_steps))
                print(q_pred.mean().item())

            if q_pred.mean().item() > 10:
                raise Exception("Stop! Grad Steps: " + str(self.grad_steps))

            elif self.grad_steps > 90000:
                raise Exception("No Crash before 60k!")

        with torch.no_grad():
            if not self.c51:
                max_actions = T.argmax(q_actions, dim=1)
                q_targets[dones] = 0.0

                q_target = rewards + (self.gamma ** self.n) * q_targets[indices, max_actions]

                if self.reward_proportions:
                    self.reward_target_avg.append(abs(float(rewards.mean().cpu())))
                    self.bootstrap_target_avg.append(abs(float(((self.gamma ** self.n) * q_targets[indices, max_actions]).mean().cpu())))
            else:

                next_actions_v = action_qvals_v.max(1)[1]

                next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
                next_best_distr_v = self.tgt_net.apply_softmax(next_best_distr_v)
                next_best_distr = next_best_distr_v.data.cpu()

                proj_distr = distr_projection(next_best_distr, rewards.cpu(), dones.cpu(), self.Vmin, self.Vmax,
                                              self.N_ATOMS,
                                              self.gamma ** self.n)

                proj_distr_v = proj_distr.to(self.net.device)

        # Trust Region Code
        if self.trust_regions:
            with torch.no_grad():

                if not self.c51:
                    losses = torch.abs(q_target - q_pred)
                else:
                    losses = (-state_log_sm_v * proj_distr_v).sum(dim=1)

                if self.running_std != -999:
                    current_std = torch.std(losses).item()
                    self.running_std += current_std

                    if self.trust_region_disable and self.grad_steps < self.trust_region_start:
                        pass
                    else:
                        if not self.c51:
                            target_network_pred = self.tgt_net.forward(states)[indices, actions]
                        else:
                            max_actions = T.argmax(action_qvals_v, dim=1)
                            next_distr_v, next_qvals_v = self.tgt_net.both(states)
                            target_network_pred = next_qvals_v[indices, max_actions]
                            q_target = rewards + (self.gamma ** self.n) * next_qvals_v[indices, max_actions]

                        sigma_j = self.running_std / self.grad_steps

                        sigma_j = max(sigma_j, current_std)
                        sigma_j = max(sigma_j, 0.01)

                        #print("sigma_j")
                        #print(sigma_j)

                        #print("Loss online to target")
                        #print(q_pred - target_network_pred)

                        outside_region = torch.abs(q_pred[indices, actions] - target_network_pred) > self.trust_alpha * sigma_j

                        diff_sign = torch.sign(q_pred[indices, actions] - target_network_pred) != torch.sign(q_pred[indices, actions] - q_target)

                        mask = torch.logical_and(outside_region, diff_sign)
                        #if self.grad_steps % 50 == 0:
                        #print(mask)

                        q_pred[mask] = 0
                        q_target[mask] = 0

                else:
                    self.running_std = torch.std(losses).detach().cpu()


        if self.my_trust_region:
            with torch.no_grad():

                #print("OG Q_targets")
                #print(q_target)
                target_network_pred = self.tgt_net.forward(states)

                mask_1 = torch.where(q_target > q_pred, 1.0, 0.0)
                mask_1 = torch.logical_and(mask_1, torch.where(q_pred > target_network_pred, 1.0, 0.0))

                mask_2 = torch.where(target_network_pred > q_pred, 1.0, 0.0)
                mask_2 = torch.logical_and(mask_2, torch.where(q_pred > q_target, 1.0, 0.0))

                mask = torch.logical_or(mask_1, mask_2)
                q_target[mask] = q_target[mask] * self.trust_alpha + q_pred[mask] * (1 - self.trust_alpha)
                #print("New Q-targets")
                #print(q_target)

        if self.my_trust_region2:
            with torch.no_grad():

                if not self.c51:
                    losses = torch.abs(q_target - q_pred)
                else:
                    losses = (-state_log_sm_v * proj_distr_v).sum(dim=1)
                    print(losses)

                if self.running_std != -999:
                    current_std = torch.std(losses).item()
                    self.running_std += current_std

                    sigma_j = self.running_std / self.grad_steps

                    sigma_j = max(sigma_j, current_std)
                    sigma_j = max(sigma_j, 0.01)
                    #print("OG Q_targets")
                    #print(q_target)


                    #self.running_std += current_std
                    #sigma_j = self.running_std / self.grad_steps
                    #self.running_std = current_std * self.std_tau + (1 - self.std_tau) * self.running_std

                    target_network_pred = self.tgt_net.forward(states)[indices, actions]

                    #sigma_j = max(0.01, self.running_std)


                    #print("sigma_j")
                    #print(sigma_j)

                    # print("Loss online to target")
                    # print(q_pred - target_network_pred)

                    outside_region = torch.abs(q_pred - target_network_pred) > sigma_j
                    diff_sign = torch.sign(q_pred - target_network_pred) != torch.sign(q_pred - q_target)
                    mask = torch.logical_and(outside_region, diff_sign)
                    # if self.grad_steps % 50 == 0:
                    #print("Mask")
                    #print(mask)

                    beta = (1 + (torch.abs(q_pred[mask] - target_network_pred[mask]) - sigma_j) / sigma_j) ** -2

                    #print("Betas")
                    #print(beta)

                    q_target[mask] = q_target[mask] * beta + q_pred[mask] * (1 - beta)
                    #q_pred[mask] = 0
                    #q_target[mask] = 0
                    #print("New Q-targets")
                    #print(q_target)

                else:
                    self.running_std = torch.std(losses).detach().cpu()

        # Loss Calculations
        if self.c51:
            loss_v = (-state_log_sm_v * proj_distr_v)

            if self.per:
                weights = T.squeeze(weights)
                loss_v = weights.to(self.net.device) * loss_v.sum(dim=1)
            else:
                loss_v = loss_v.sum(dim=1)

            if self.variance_data:
                self.variances.append(torch.var(loss_v).item())

            loss = loss_v.mean()
        else:

            if not self.per:
                loss = self.net.loss(q_target, q_pred).to(self.net.device)
                if self.variance_data:
                    self.variances.append(torch.var(loss).item())
            else:
                td_error = q_target - q_pred
                loss_v = (td_error.pow(2)*weights.to(self.net.device))
                loss = loss_v.mean().to(self.net.device)
                if self.variance_data:
                    self.variances.append(torch.var(loss_v).item())


        loss.backward()
        T.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.net.optimizer.step()

        if self.per:
            if self.c51:
                self.memory.update_priorities(idxs, abs(loss_v.cpu().detach().numpy()))
            else:
                self.memory.update_priorities(idxs, abs(td_error.cpu().detach().numpy()))

        self.grad_steps += 1

        self.epsilon.update_eps()

        if self.variance_data and self.grad_steps == 97000 - self.min_sampling_size:
            variance_data = np.array(self.variances)
            np.save(self.algo_name + "_varianceData" + self.game + str(self.run) + ".npy", variance_data)


        if self.action_gap_data:
            #print(q_pred_og)
            #print(torch.topk(q_pred_og, 2).values)
            #print(q_pred_og.shape)
            #print(torch.topk(q_pred_og, 2).values.shape)

            top2 = torch.topk(q_pred_og, 2).values
            first_column = top2[:, 0]
            second_column = top2[:, 1]

            # Calculate the difference between the first and second columns
            result_tensor = first_column - second_column

            avg = result_tensor.mean()
            #print(avg)

            self.action_gaps.append(avg.item())

            if self.grad_steps % 500 == 0 and self.grad_steps > 80000:
                action_gaps = np.array(self.action_gaps)
                np.save(self.algo_name + "_actionGaps" + self.game + str(self.run) + ".npy", action_gaps)

        if self.reward_proportions:
            if self.grad_steps % 500 == 0:
                print("\nRewards: " + str(sum(self.reward_target_avg) / self.grad_steps))
                print("Bootstraps: " + str(sum(self.bootstrap_target_avg) / self.grad_steps))

                """length = len(self.reward_target_avg)
                x = np.arange(length)
                y = np.array(self.reward_target_avg) / (np.array(self.reward_target_avg) + np.array(self.bootstrap_target_avg))
                plt.plot(x,y)
                plt.xlabel("TimeSteps")
                plt.ylabel("Reward Proportion")
                plt.title("N=10")
                plt.show()"""

                if self.grad_steps > 1000:
                    np.save(self.algo_name + "_proportionRewards" + self.game + str(self.run) + ".npy", np.array(self.reward_target_avg))
                    np.save(self.algo_name + "_proportionBootstrap" + self.game + str(self.run) + ".npy", np.array(self.bootstrap_target_avg))

        if self.identify_data:

            self.identify.batch_Qvals.append(q_pred_og.detach().cpu().numpy())
            self.identify.batch_idxs.append(self.memory.last_batch)
            self.identify.batch_loss.append(loss.detach().cpu().item())
            self.identify.batch_target_states_vals.append(q_targets.detach().cpu().numpy())
            self.identify.batch_target_actions.append(max_actions.detach().cpu().numpy())

            new_q_vals = self.net(states)
            self.identify.batch_new_Qvals.append(new_q_vals.detach().cpu().numpy())

            # policy churn calculation
            sample_size = min(self.churn_sample, self.memory.mem_cntr - self.n)
            states, _, _, _, _ = self.memory.sample_memory(bs=sample_size)
            states = T.tensor(states).to(self.net.device)
            cur_vals = self.net(states)
            if self.replace_target_cnt > 1:
                tgt_vals = self.churn_net(states)
            else:
                tgt_vals = self.tgt_net(states)

            output = torch.argmax(cur_vals, dim=1)
            tgt_output = torch.argmax(tgt_vals, dim=1)

            policy_churn = ((sample_size - torch.sum(output == tgt_output)) / sample_size).item()
            self.identify.churn.append(policy_churn)

            # change this to be at the end
            if self.env_steps == 2500: #this value needs to be greater than min sampling size
                self.identify.er_states = self.memory.state_memory
                self.identify.er_actions = self.memory.action_memory
                self.identify.er_rewards = self.memory.reward_memory
                self.identify.er_dones = self.memory.terminal_memory
                self.identify.er_next_states = self.memory.new_state_memory

                with mgzip.open("..\\" + self.algo_name + "_" + self.game + '_identify_data.pkl', 'wb') as f:
                    pickle.dump(self.identify, f)

                raise Exception("Stop we are done here")

        if self.gen_data:

            with T.no_grad():
                q_pred = self.tgt_net(states)
                q_pred_new = self.net(states)

                """
                print("\nBefore")
                print(q_pred)
                print("After")
                print(q_pred_new)
                print("Difference")
                print(q_pred - q_pred_new)
                """

                # total change is just this step. self.total_change is the running total
                total_change = torch.abs(q_pred - q_pred_new).sum().cpu().detach().item()
                self.total_change += total_change

                A_s1 = T.clone(q_pred)
                A_s_new1 = T.clone(q_pred_new)

                A_s1[0, actions[0]] = 0
                A_s_new1[0, actions[0]] = 0

                self.total_gen_change += torch.abs(A_s1 - A_s_new1).sum().cpu().detach().item()

                for i in range(len(q_pred[0])):

                    if i != actions[0]:
                        q_pred[0, i] = 0
                        q_pred_new[0, i] = 0

                target_change = torch.abs(q_pred - q_pred_new).sum().cpu().detach().item()
                self.total_targetted_change += target_change

                tau = 0.999
                if total_change > 0:
                    new_val = target_change / total_change
                else:
                    new_val = self.target_percent_mov_avg
                self.target_percent_mov_avg = self.target_percent_mov_avg * tau + new_val * (1 - tau)
                self.target_percent_list.append(self.target_percent_mov_avg)

                if np.random.random() > 0.999:
                    print("\nTotal Change: " + str(self.total_change))
                    print("Total Generalisation Change: " + str(self.total_gen_change))
                    print("Total Targetted Change: " + str(self.total_targetted_change))
                    print("Target Percent Moving Average: " + str(self.target_percent_mov_avg))


                if self.grad_steps == 97000:
                    np_dat = np.array(self.target_percent_list,dtype=np.float32)
                    np.save(self.algo_name + '_' + self.game + str(self.run) + '_target_gen_data.npy', np_dat)

        if self.collecting_churn_data:
            if not self.reset_churn and self.env_steps > self.start_churn + self.churn_dur:
                self.reset_churn = True

                # save data
                self.save_churn_data()
                self.save_churn_data()

                self.total_churn = 0
                self.churn_data = []
                self.churn_actions = np.array([0 for i in range(self.n_actions)], dtype=np.float64)
                self.count_since_reset = 0
                self.action_swaps = np.zeros((self.n_actions, self.n_actions), dtype=np.int64)

            if not self.second_save and self.env_steps > self.second_churn + self.churn_dur:
                self.second_save = True
                # save data
                self.save_churn_data()

                self.total_churn = 0
                self.churn_data = []
                self.churn_actions = np.array([0 for i in range(self.n_actions)], dtype=np.float64)
                self.count_since_reset = 0

            if self.start_churn < self.env_steps < self.start_churn + self.churn_dur or \
                    self.second_churn < self.env_steps < self.second_churn + self.churn_dur:

                self.collect_churn_data()
                self.count_since_reset += 1


    def save_churn_data(self):
        avg_churn = self.total_churn / self.count_since_reset
        median_churn = np.percentile(self.churn_data, 50)
        per90 = np.percentile(self.churn_data, 90)
        per99 = np.percentile(self.churn_data, 99)
        per99_9 = np.percentile(self.churn_data, 99.9)
        churns_per_action = self.churn_actions
        percent_churns_per_actions = self.churn_actions / np.sum(self.churn_actions)
        total_action_percents = self.total_actions / np.sum(self.total_actions)
        churn_std = np.std(percent_churns_per_actions)
        action_std = np.std(total_action_percents)

        x = torch.FloatTensor(self.churn_data)
        x = torch.topk(x, 50).values
        top50churns = []
        for i in x:
            top50churns.append(i.item())

        game = self.game
        if not self.second_save:
            start_timesteps = self.start_churn
            end_timesteps = self.start_churn + self.churn_dur
        else:
            start_timesteps = self.second_churn
            end_timesteps = self.second_churn + self.churn_dur

        percent0churn = self.churn_data.count(0.) / len(self.churn_data)

        churn_data = ChurnData(avg_churn, per90, per99, per99_9, churns_per_action, percent_churns_per_actions,
                               total_action_percents,churn_std, action_std, top50churns, game, start_timesteps,
                               end_timesteps, percent0churn, self.algo_name, median_churn, self.action_swaps)

        with open(self.algo_name + "_" + game + str(start_timesteps) + "_" + str(self.run) + '.pkl', 'wb') as outp:
            pickle.dump(churn_data, outp, pickle.HIGHEST_PROTOCOL)

    def collect_churn_data(self):
        sample_size = min(self.churn_sample, self.env_steps - self.n) - (min(self.churn_sample, self.env_steps - self.n) % self.batch_size)
        if not self.per:

            states, _, _, _, _ = self.memory.sample_memory(bs=sample_size)
        else:

            _, states, _, _, _, _, _ = self.memory.sample(self.batch_size)
            if states is False:
                return
            for i in range(math.floor(min(self.churn_sample, self.env_steps - self.n) / self.batch_size)):
                _, statesX, _, _, _, _, _ = self.memory.sample(self.batch_size)
                if statesX is False:
                    return
                states = torch.cat((states, statesX))

        states = T.tensor(states).to(self.net.device)

        if self.noisy:
            self.net.disable_noise()
            self.churn_net.disable_noise()

        if not self.c51:
            cur_vals = self.net(states)

            if self.replace_target_cnt > 1 or self.target_ema or self.noisy:
                tgt_vals = self.churn_net(states)
            else:
                tgt_vals = self.tgt_net(states)
        else:
            cur_vals = self.net.qvals(states)

            if self.replace_target_cnt > 1 or self.target_ema or self.noisy:
                tgt_vals = self.churn_net.qvals(states)
            else:
                tgt_vals = self.tgt_net.qvals(states)

        if self.noisy:
            self.net.enable_noise()
            self.churn_net.enable_noise()

        output = torch.argmax(cur_vals, dim=1)
        tgt_output = torch.argmax(tgt_vals, dim=1)

        policy_churn = ((sample_size - torch.sum(output == tgt_output)) / sample_size).item()
        self.total_churn += policy_churn
        self.churn_data.append(policy_churn)

        dif = torch.abs(torch.subtract(cur_vals, tgt_vals))
        dif = torch.sum(dif, dim=0).detach().cpu().numpy()

        self.churn_actions += dif

        changes = output != tgt_output
        output = output[changes]
        tgt_output = tgt_output[changes]

        for i in range(len(output)):
            self.action_swaps[output[i], tgt_output[i]] += 1

        """
        if np.random.random() > 0.99 and len(self.churn_data) > 100:
            percent_actions = self.churn_actions / np.sum(self.churn_actions)

            print("\n\n")
            print("Avg churn: " + str(self.total_churn / self.count_since_reset))
            print("90th per: " + str(np.percentile(self.churn_data, 90)))
            print("99th per: " + str(np.percentile(self.churn_data, 99)))
            print("99.9th per: " + str(np.percentile(self.churn_data, 99.9)))

            x = torch.FloatTensor(self.churn_data)
            x = torch.topk(x, 50).values
            temp = []
            for i in x:
                temp.append(i.item())
            print("Top Churns: " + str(temp))

            print("Percentages of churn by action: " + str(percent_actions))
            print("Portions of actions taken: " + str(self.total_actions / np.sum(self.total_actions)))

            print("std churn: " + str(np.std(percent_actions)))
            print("std actions taken: " + str(np.std(self.total_actions / np.sum(self.total_actions))))

            #print(self.churn_data)
            print("Percent 0 Churn: " + str(self.churn_data.count(0.) / len(self.churn_data)))

            print("Action Swap Matrix: \n" + str(self.action_swaps))
        """


def running_average_with_window(input_list, window_size):
    if window_size <= 0:
        raise ValueError("Window size should be a positive integer.")

    averages = []
    window = deque(maxlen=window_size)

    for value in input_list:
        window.append(value)
        current_average = sum(window) / len(window)
        averages.append(current_average)

    return averages



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