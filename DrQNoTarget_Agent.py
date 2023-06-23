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

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1V = nn.Linear(64 * 7 * 7, 512)
        self.fc1A = nn.Linear(64 * 7 * 7, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
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
                 max_mem_size=100000,total_frames=100000,lr=0.0001,batch_size=32,discount=0.99):

        self.epsilon = EpsilonGreedy()
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.min_sampling_size = 1600
        self.n = 10
        self.chkpt_dir = ""
        self.gamma = discount
        self.eval_mode = False
        self.grad_steps = 1

        self.memory = ExperienceReplay(input_dims, max_mem_size, self.batch_size)

        self.net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_eval',
                                          chkpt_dir=self.chkpt_dir, device=device)

        self.n_states = deque([], self.n)
        self.n_rewards = deque([], self.n)
        self.n_actions = deque([], self.n)

        self.random_shift = nn.Sequential(nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
        self.intensity = Intensity(scale=0.05)

    def get_grad_steps(self):
        return self.grad_steps

    def choose_action(self, observation):
        if np.random.random() > self.epsilon.eps or self.eval_mode:
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
            self.memory.store_transition(self.n_states[0], self.n_actions[0], fin_reward,
                                         state_, done)

        if done:
            self.n_states = deque([], self.n)
            self.n_rewards = deque([], self.n)
            self.n_actions = deque([], self.n)


    def save_models(self):
        self.net.save_checkpoint()

    def load_models(self):
        self.net.load_checkpoint()

    def learn(self):

        if self.memory.mem_cntr < self.min_sampling_size:
            return

        self.net.optimizer.zero_grad()

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

        all_states = T.cat((states_aug, states_aug_, states_aug_policy_))
        Vs, As = self.net.forward(all_states)

        V_s, V_s_, V_s_eval = T.tensor_split(Vs, 3, dim=0)
        A_s, A_s_, A_s_eval = T.tensor_split(As, 3, dim=0)

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


