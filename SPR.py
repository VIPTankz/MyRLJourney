import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memorySPR import ReplayMemory
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

  def forward(self, input, no_noise=False):
    if self.training and not no_noise:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class C51(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir,atoms,Vmax,Vmin, device):
        super(C51, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.atoms = atoms
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.DELTA_Z = (self.Vmax - self.Vmin) / (self.atoms - 1)
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1V = NoisyLinear(64 * 7 * 7, 256)
        self.fc1A = NoisyLinear(64 * 7 * 7, 256)
        self.fcV2 = NoisyLinear(256, atoms)
        self.fcA2 = NoisyLinear(256, n_actions * atoms)

        self.prediction_head = nn.Linear(512, 512)

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

    def prediction(self, x):
        # input here should be the flattened output of the transition model

        v_latent = F.relu(self.fc1V(x, no_noise=True))
        a_latent = F.relu(self.fc1A(x, no_noise=True))

        latent = torch.cat((v_latent, a_latent), 1)

        pred = self.prediction_head(latent)

        return pred

    def target_prediction(self, x):
        #input here is just the state

        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)

        v_latent = F.relu(self.fc1V(conv_out, no_noise=True))
        a_latent = F.relu(self.fc1A(conv_out, no_noise=True))

        latent = torch.cat((v_latent, a_latent), 1)

        return latent


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


class TransitionModel(nn.Module):
    def __init__(self, n_actions, device):
        super(TransitionModel, self).__init__()
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(64 + n_actions, 64, 3, padding=1)
        self.batch_norm = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        self.to(device)

    def forward(self, x, action):
        # takes z, (the output of the encoder or the output of this model)
        # create the action onehot vector
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0], self.n_actions,
                                    x.shape[-2], x.shape[-1], device=action.device)
        action_onehot[batch_range, action, :, :] = 1

        stacked_x = torch.cat([x, action_onehot], 1)

        x = F.relu(self.conv1(stacked_x))
        x = self.batch_norm(x)

        x = F.relu(self.conv2(x))

        # output should be batch_size*64*7*7
        return x

class Agent():
    def __init__(self, n_actions, input_dims, device,
                 max_mem_size=100000, total_frames=100000, lr=0.0001,
                 game=None, run=None, name=None):

        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims

        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.min_sampling_size = 2000

        self.chkpt_dir = ""

        self.run = run
        self.algo_name = name
        self.game = game

        #MAKE SURE YOU CHECKED THE NAME AND DATA COLLETION

        # IMPORTANT params, check these
        self.n = 10
        self.gamma = 0.99
        self.batch_size = 32 #CHANGED
        self.replace_target_cnt = 1
        self.replay_ratio = 2
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

        # c51
        self.Vmax = 10
        self.Vmin = -10
        self.N_ATOMS = 51

        # spr loss
        self.K = 5
        self.spr_loss_coef = 2

        # augmentations
        self.random_shift = nn.Sequential(nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
        self.intensity = Intensity(scale=0.05)

        self.memory = ReplayMemory(max_mem_size, self.n, self.gamma, device, self.K)

        self.net = C51(self.lr, self.n_actions,
                                          input_dims=self.input_dims, name='DER_eval',
                                          chkpt_dir=self.chkpt_dir,atoms=self.N_ATOMS,Vmax=self.Vmax,Vmin=self.Vmin, device=device)

        self.tgt_net = C51(self.lr, self.n_actions,
                                          input_dims=self.input_dims,name='DER_next',
                                          chkpt_dir=self.chkpt_dir,atoms=self.N_ATOMS,Vmax=self.Vmax,Vmin=self.Vmin, device=device)

        self.transition_model = TransitionModel(self.n_actions, device)

        self.net.train()
        self.tgt_net.train()

        for param in self.tgt_net.parameters():
            param.requires_grad = False

        self.env_steps = 0
        self.grad_steps = 0
        self.reset_churn = False
        self.second_save = False

        self.epsilon = 0.001 #used for eval only

        self.game = game

        self.replay_ratio_cnt = 0
        self.eval_mode = False

        self.priority_weight_increase = (1 - 0.4) / (total_frames - self.min_sampling_size)

        self.cosine = torch.nn.CosineSimilarity(dim=2)

    def produce_latents(self, x, actions):
        # takes states as input

        # first pass through CNN
        x = x.float() / 256
        z = self.net.conv(x)

        latents = []

        for i in range(self.K):

            z = self.transition_model(z, actions[i])
            bs = len(z)
            latent = self.net.prediction(torch.reshape(z, (bs, -1)))
            latents.append(latent)

        latents = torch.cat(latents, 0)
        return latents

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

        states = (self.intensity(self.random_shift(states.float()))).to(T.uint8)
        states_ = (self.intensity(self.random_shift(states_.float()))).to(T.uint8)
        states_policy_ = (self.intensity(self.random_shift(states_.float()))).to(T.uint8)

        ############ Rainbow Loss
        distr_v, qvals_v = self.net.both(states)
        state_action_values = distr_v[range(self.batch_size), actions.data]
        state_log_sm_v = F.log_softmax(state_action_values, dim=1)

        with torch.no_grad():
            self.tgt_net.reset_noise()
            next_distr_v, next_qvals_v = self.tgt_net.both(states_)
            action_distr_v, action_qvals_v = self.net.both(states_policy_)

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

        rainbow_loss = loss_v.mean()
        ############################
        # SPR Loss

        # fetch tgt states and actions
        tgt_states, prediction_actions, prediction_terminals = self.memory.get_spr_sample()
        tgt_states = tgt_states.clone().detach().to(self.net.device)
        prediction_actions = prediction_actions.clone().detach().to(self.net.device)
        prediction_terminals = prediction_terminals.clone().detach().to(self.net.device).squeeze()

        tgt_states = torch.reshape(tgt_states, (self.K * self.batch_size, 4, 84, 84))

        # perform augmentations on target states
        tgt_states = (self.intensity(self.random_shift(tgt_states.float()))).to(T.uint8)

        # Calculate target latents
        tgt_latents = self.tgt_net.target_prediction(tgt_states)

        # Calculate online latents
        latents = self.produce_latents(states, prediction_actions)

        # do OR operations on terminals
        prediction_terminals = process_tensor(prediction_terminals)

        # mask out terminals
        prediction_terminals = prediction_terminals.flatten().unsqueeze(1)
        latents = latents * prediction_terminals
        tgt_latents = tgt_latents * prediction_terminals

        # reshape back to K,batch size,512
        latents = torch.reshape(latents, (self.K, self.batch_size, 512))
        tgt_latents = torch.reshape(tgt_latents, (self.K, self.batch_size, 512))

        # latents and tgt_latents are K,batch_size,512 and we want to get sum of differences in the last dim
        spr_loss = ((self.cosine(latents, tgt_latents).sum()) * -1) / self.batch_size

        ################
        final_loss = self.spr_loss_coef * spr_loss + rainbow_loss
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.net.optimizer.step()

        self.grad_steps += 1

        self.memory.update_priorities(idxs, loss_v.cpu().detach().numpy())


def process_tensor(tensor):
    for col in range(tensor.shape[1]):
        col_values = tensor[:, col]
        zero_indices = torch.nonzero(col_values == 0)
        if zero_indices.numel() > 0:
            first_zero_row = zero_indices[0, 0].item()
            tensor[first_zero_row:, col] = 0
    return tensor

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