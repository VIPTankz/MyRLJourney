import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from torchvision.utils import save_image
import math

from ChurnData import ChurnData
from PrioritisedExperienceReplay import PrioritizedReplayBuffer
import kornia.augmentation as aug
import pickle
import torch

class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = T.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise

class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.5, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(T.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_input", T.zeros(1, in_features))
        self.register_buffer("epsilon_output", T.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(T.full((out_features,), sigma_init))

    def forward(self, input, use_noise=True):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: T.sign(x) * T.sqrt(T.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = T.mul(eps_in, eps_out)

        if use_noise:
            return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)
        else:
            return F.linear(input, self.weight)

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

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1V = NoisyFactorizedLinear(64 * 7 * 7, 512)
        self.fc1A = NoisyFactorizedLinear(64 * 7 * 7, 512)
        self.V = NoisyFactorizedLinear(512, atoms)
        self.A = NoisyFactorizedLinear(512, n_actions * atoms)

        self.register_buffer("supports", T.arange(Vmin, Vmax + self.DELTA_Z, self.DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = device
        print("Device: " + str(self.device),flush=True)
        self.to(self.device)

    def reset_mlp(self):
        self.fc1V = NoisyFactorizedLinear(64 * 7 * 7, 512)
        self.fc1A = NoisyFactorizedLinear(64 * 7 * 7, 512)
        self.V = NoisyFactorizedLinear(512, self.atoms)
        self.A = NoisyFactorizedLinear(512, self.n_actions * self.atoms)
        self.to(self.device)

    def conv(self,x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x

    def fc_val(self, x, use_noise=True):
        x = F.relu(self.fc1V(x, use_noise))
        x = self.V(x, use_noise)

        return x

    def fc_adv(self, x, use_noise=True):
        x = F.relu(self.fc1A(x, use_noise))
        x = self.A(x, use_noise)

        return x

    def forward(self, x, use_noise=True):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out, use_noise).view(batch_size, 1, self.atoms)
        adv_out = self.fc_adv(conv_out, use_noise).view(batch_size, -1, self.atoms)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def both(self, x, use_noise=True):
        cat_out = self(x, use_noise)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x, use_noise=True):
        return self.both(x, use_noise)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, self.atoms)).view(t.size())

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
    def __init__(self, n_actions,input_dims, device,
                 max_mem_size=100000, replace=1,total_frames=100000,lr=0.0001,batch_size=32,discount=0.99,
                 game=None, run=None):

        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.min_sampling_size = 2000
        self.chkpt_dir = ""
        self.gamma = discount
        self.eval_mode = False
        self.grad_steps = 2

        self.run = run
        self.algo_name = "DrDER_small_churn"

        #n-step
        self.n = 10
        self.nstep_states = deque([], self.n)
        self.nstep_rewards = deque([], self.n)
        self.nstep_actions = deque([], self.n)

        #c51
        self.Vmax = 10
        self.Vmin = -10
        self.N_ATOMS = 51

        self.resets = False
        self.reset_times = [40000, 75000]

        self.memory = PrioritizedReplayBuffer(input_dims, n_actions, max_mem_size, eps=1e-5, alpha=0.5, beta=0.4,
                                                  total_frames=total_frames)

        self.net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims, name='DER_eval',
                                          chkpt_dir=self.chkpt_dir,atoms=self.N_ATOMS,Vmax=self.Vmax,Vmin=self.Vmin, device=device)

        self.tgt_net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,name='DER_next',
                                          chkpt_dir=self.chkpt_dir,atoms=self.N_ATOMS,Vmax=self.Vmax,Vmin=self.Vmin, device=device)

        self.random_shift = nn.Sequential(nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
        self.intensity = Intensity(scale=0.05)

        self.collecting_churn_data = True

        self.env_steps = 0
        self.reset_churn = False
        self.second_save = False

        self.start_churn = self.min_sampling_size
        self.churn_sample = 10000
        self.churn_dur = 23000
        self.second_churn = 75000
        self.total_churn = 0
        self.churn_data = []
        self.churn_actions = np.array([0 for i in range(self.n_actions)], dtype=np.float64)
        self.total_actions = np.array([0 for i in range(self.n_actions)], dtype=np.int64)
        self.count_since_reset = 0
        self.game = game

        self.action_swaps = np.zeros((self.n_actions, self.n_actions), dtype=np.int64)

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.net.device)
        with T.no_grad():
            advantage = self.net.qvals(state)
            x = T.argmax(advantage).item()
        return x

    def get_grad_steps(self):
        return self.grad_steps

    def set_eval_mode(self):
        pass

    def store_transition(self, state, action, reward, state_, done):
        self.n_step(state, action, reward, state_, done)
        self.env_steps += 1
        self.total_actions[action] += 1

    def n_step(self, state, action, reward, state_, done):
        self.nstep_states.append(state)
        self.nstep_rewards.append(reward)
        self.nstep_actions.append(action)

        if len(self.nstep_states) == self.n:
            fin_reward = 0
            for i in range(self.n):
                fin_reward += self.nstep_rewards[i] * (self.gamma ** i)
            self.memory.add(self.nstep_states[0], self.nstep_actions[0],fin_reward, state_, done)

        if done:
            self.nstep_states = deque([], self.n)
            self.nstep_rewards = deque([], self.n)
            self.nstep_actions = deque([], self.n)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.tgt_net.load_state_dict(self.net.state_dict())

    def save_models(self):
        self.net.save_checkpoint()
        self.tgt_net.save_checkpoint()

    def load_models(self):
        self.net.load_checkpoint()
        self.tgt_net.load_checkpoint()

    def learn(self):
        for i in range(self.grad_steps):
            self.learn_call()

    def learn_call(self):

        if self.memory.count < self.min_sampling_size:
            return

        self.net.optimizer.zero_grad()

        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.replace_target_network()

        if self.resets:
            if self.env_steps in self.reset_times:
                self.net.reset_mlp()

        batch, weights, tree_idxs = self.memory.sample(self.batch_size)

        states, actions, rewards, new_states, dones = batch

        states = states.to(self.net.device)
        rewards = rewards.to(self.net.device)
        dones = dones.to(self.net.device)
        actions = actions.to(self.net.device)
        states_ = new_states.to(self.net.device)
        weights = weights.to(self.net.device)

        ############## Data Augmentation

        states = (self.intensity(self.random_shift(states.float()/255.)) * 255).to(T.uint8)
        states_ = (self.intensity(self.random_shift(states_.float()/255.)) * 255).to(T.uint8)
        states_policy_ = (self.intensity(self.random_shift(states_.float()/255.)) * 255).to(T.uint8)

        ##############

        distr_v, qvals_v = self.net.both(T.cat((states, states_policy_)))
        next_qvals_v = qvals_v[self.batch_size:]
        distr_v = distr_v[:self.batch_size]

        next_distr_v, _ = self.tgt_net.both(states_)

        next_actions_v = next_qvals_v.max(1)[1]
        next_distr_v = self.tgt_net(states_)
        next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
        next_best_distr_v = self.tgt_net.apply_softmax(next_best_distr_v)
        next_best_distr = next_best_distr_v.data.cpu()

        proj_distr = distr_projection(next_best_distr, rewards.cpu(), dones.cpu(), self.Vmin, self.Vmax, self.N_ATOMS, self.gamma)

        state_action_values = distr_v[range(self.batch_size), actions.data]
        state_log_sm_v = F.log_softmax(state_action_values, dim=1)
        proj_distr_v = proj_distr.to(self.net.device)

        loss_v = -state_log_sm_v * proj_distr_v
        weights = T.squeeze(weights)
        loss_v = weights.to(self.net.device) * loss_v.sum(dim=1)

        loss = loss_v.mean()
        ##############

        loss.backward()
        T.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.net.optimizer.step()
        self.learn_step_counter += 1

        self.memory.update_priorities(tree_idxs, loss_v.cpu().detach().numpy())

        if self.collecting_churn_data:
            if not self.reset_churn and self.env_steps > self.start_churn + self.churn_dur:
                self.reset_churn = True

                # save data
                self.save_churn_data()

                self.total_churn = 0
                self.churn_data = []
                self.churn_actions = np.array([0 for i in range(self.n_actions)], dtype=np.float64)
                self.count_since_reset = 0

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
        sample_size = min(self.churn_sample, self.memory.count)
        batch, _, _ = self.memory.sample(sample_size)
        states, _, _, _, _ = batch

        states = states.to(self.net.device)

        cur_vals = self.net.qvals(states, use_noise=False)
        tgt_vals = self.tgt_net.qvals(states, use_noise=False)

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

        if np.random.random() > 0.9 and len(self.churn_data) > 100:
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

            raise Exception("stop")


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

