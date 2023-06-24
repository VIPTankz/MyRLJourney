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

        self.memory = ExperienceReplay(input_dims, max_mem_size, self.batch_size)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_eval',
                                          chkpt_dir=self.chkpt_dir, device=device)

        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
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
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
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
            self.memory.store_transition(self.n_states[0], self.n_actions[0], fin_reward, \
                                         state_, done)

        if done:
            self.n_states = deque([], self.n)
            self.n_rewards = deque([], self.n)
            self.n_actions = deque([], self.n)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):

        if self.memory.mem_cntr < self.min_sampling_size:
            return

        self.q_eval.optimizer.zero_grad()

        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.replace_target_network()

        states, actions, rewards, new_states, dones = self.memory.sample_memory()

        states = T.tensor(states).to(self.q_eval.device)
        rewards = T.tensor(rewards).to(self.q_eval.device)
        dones = T.tensor(dones).to(self.q_eval.device)
        actions = T.tensor(actions).to(self.q_eval.device)
        states_ = T.tensor(new_states).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        states_aug = (self.intensity(self.random_shift(states.float()/255.)) * 255).to(T.uint8)
        states_aug_ = (self.intensity(self.random_shift(states_.float()/255.)) * 255).to(T.uint8)
        states_aug_policy_ = (self.intensity(self.random_shift(states_.float()/255.)) * 255).to(T.uint8)

        V_s, A_s = self.q_eval.forward(states_aug)

        V_s_, A_s_ = self.q_next.forward(states_aug_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_aug_policy_)

        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                       (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + (self.gamma ** self.n) * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        loss.backward()
        T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 10)
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.epsilon.update_eps()


