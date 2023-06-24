import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ExperienceReplay import ExperienceReplay
import numpy as np
from collections import deque
import kornia.augmentation as aug

class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = T.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, device):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1V = nn.Linear(64 * 7 * 7, 512)
        self.fc1A = nn.Linear(64 * 7 * 7, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=0.00015)
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def forward(self, observation):
        observation = T.div(observation, 255)
        observation = observation.view(-1, 4, 84, 84)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 64 * 7 * 7)
        observationV = F.relu(self.fc1V(observation))
        observationA = F.relu(self.fc1A(observation))
        V = self.V(observationV)
        A = self.A(observationA)

        return V, A

    def reset_mlps(self):
        self.fc1V = nn.Linear(64 * 7 * 7, 512)
        self.fc1A = nn.Linear(64 * 7 * 7, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, self.n_actions)

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
                 max_mem_size=100000, replace=1,total_frames=100000,lr=0.0001,batch_size=32,discount=0.99):

        self.epsilon = EpsilonGreedy()
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.min_sampling_size = 1600
        self.n = 10
        self.chkpt_dir = ""
        self.gamma = discount
        self.grad_steps = 1

        self.device = device

        self.memory = ExperienceReplay(input_dims, max_mem_size, self.batch_size)

        # Resetting Parameters
        self.reset_layers_every = 40000
        self.conv_move_amount = 0.8
        self.reset_conv_layers = False

        self.net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_eval',
                                          chkpt_dir=self.chkpt_dir, device=device)

        self.tgt_net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_next',
                                          chkpt_dir=self.chkpt_dir, device=device)

        self.n_states = deque([], self.n)
        self.n_rewards = deque([], self.n)
        self.n_actions = deque([], self.n)

        self.random_shift = nn.Sequential(nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
        self.intensity = Intensity(scale=0.05)

    def get_grad_steps(self):
        return self.grad_steps

    def set_eval_mode(self):
        self.epsilon.eps_final = 0.05
        self.epsilon = 0.05

    def choose_action(self, observation):
        if np.random.random() > self.epsilon.eps:
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.net.device)
            _, advantage = self.net.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.n_step(state, action, reward, state_, done)

    def n_step(self, state, action, reward, state_, done):
        self.n_states.append(state)
        self.n_rewards.append(reward)
        self.n_actions.append(action)

        if len(self.n_states) == self.n:
            fin_reward = 0
            for i in range(self.n):
                fin_reward += self.n_rewards[i] * (self.gamma ** i)
            self.memory.store_transition(self.n_states[0], self.n_actions[0], fin_reward, state_, done)

        if done:
            self.n_states = deque([], self.n)
            self.n_rewards = deque([], self.n)
            self.n_actions = deque([], self.n)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.tgt_net.load_state_dict(self.net.state_dict())

    def save_models(self):
        self.net.save_checkpoint()
        self.tgt_net.save_checkpoint()

    def load_models(self):
        self.net.load_checkpoint()
        self.tgt_net.load_checkpoint()

    def shrink_and_perturb(self):
        random_model = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_eval',
                                          chkpt_dir=self.chkpt_dir, device=self.device)

        params1 = self.net.named_parameters()
        params2 = random_model.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(self.conv_move_amount * param1.data + (1 - self.conv_move_amount) *
                                               dict_params2[name1].data)

        self.net.load_state_dict(dict_params2)
        self.net.to(self.device)

    def learn(self):
        for i in range(self.grad_steps):
            self.learn_call()

    def learn_call(self):

        if self.memory.mem_cntr < self.min_sampling_size:
            return

        self.net.optimizer.zero_grad()

        if self.learn_step_counter % self.reset_layers_every == 0:
            if self.reset_conv_layers:
                self.shrink_and_perturb()

            self.net.reset_mlps()

            self.net.to(self.device)

        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.replace_target_network()

        states, actions, rewards, new_states, dones = self.memory.sample_memory()

        states = T.tensor(states).to(self.net.device)
        rewards = T.tensor(rewards).to(self.net.device)
        dones = T.tensor(dones).to(self.net.device)
        actions = T.tensor(actions).to(self.net.device)
        states_ = T.tensor(new_states).to(self.net.device)

        indices = np.arange(self.batch_size)

        states_aug = (self.intensity(self.random_shift(states.float() / 255.)) * 255.).to(T.uint8)
        states_aug_ = (self.intensity(self.random_shift(states_.float() / 255.)) * 255.).to(T.uint8)
        states_aug_policy_ = (self.intensity(self.random_shift(states_.float() / 255.)) * 255.).to(T.uint8)

        net_batch = T.cat((states_aug, states_aug_policy_))

        # just batches these together to speed up computation
        Vs, As = self.net.forward(net_batch)

        V_s, V_s_eval = T.tensor_split(Vs, 2, dim=0)
        A_s, A_s_eval = T.tensor_split(As, 2, dim=0)

        # even though there isn't really a target network, we still use two to avoid grad issues
        # perhaps you could do with T.no_grad()? Might save some memory and copying
        V_s_, A_s_ = self.tgt_net.forward(states_aug_)

        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                       (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + (self.gamma ** self.n) * q_next[indices, max_actions]

        loss = self.net.loss(q_target, q_pred).to(self.net.device)

        loss.backward()
        T.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.net.optimizer.step()
        self.learn_step_counter += 1

        self.epsilon.update_eps()


