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

class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = T.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class HydraQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, device):
        super(HydraQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1s = nn.ParameterList([nn.Linear(64 * 7 * 7, 512) for i in range(n_actions)])
        self.Qs = nn.ParameterList([nn.Linear(512, 1) for i in range(n_actions)])

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
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

        A = []
        for i in range(self.n_actions):
            fc1 = F.relu(self.fc1s[i](observation))
            A.append(self.Qs[i](fc1))

        Q = torch.stack(A, dim=1).squeeze(dim=-1)

        return Q

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class LeggedQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, device):
        super(LeggedQNetwork, self).__init__()

        self.n_actions = n_actions

        self.conv1 = nn.ParameterList([nn.Conv2d(4, 32, 8, stride=4, padding=1) for i in range(n_actions)])
        #self.conv2 = nn.ParameterList([nn.Conv2d(32, 64, 4, stride=2) for i in range(n_actions)])
        #self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.ParameterList([nn.Conv2d(32, 64, 4, stride=2) for i in range(n_actions)])
        #self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.ParameterList([nn.Conv2d(64, 64, 3) for i in range(n_actions)])

        self.fc1s = nn.Linear(64 * 7 * 7, 512)
        self.Qs = nn.Linear(512, n_actions)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def forward(self, observation):
        observation = T.div(observation, 255)
        observation = observation.view(-1, 4, 84, 84)

        A = []
        for i in range(self.n_actions):
            observationI = F.relu(self.conv1[i](observation))
            observationI = F.relu(self.conv2[i](observationI))
            observationI = F.relu(self.conv3[i](observationI))
            observationI = observationI.view(-1, 64 * 7 * 7)

            fc1 = F.relu(self.fc1s(observationI))
            A.append(self.Qs(fc1)[:, i])

        Q = torch.stack(A, dim=1)

        return Q

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class SanityNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, device):
        super(SanityNetwork, self).__init__()
        self.n_actions = n_actions

        #self.V_network = ActionNetwork()
        self.action_networks = nn.ParameterList([ActionNetwork() for i in range(self.n_actions)])

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        #self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=0.00015)
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def forward(self, observation):
        outs = []

        for i in self.action_networks:
            outs.append(i(observation))

        Q = torch.stack(outs, dim=1)
        #V = self.V_network(observation)

        #Q = V + (A - A.mean(dim=1, keepdim=True)).squeeze(dim=-1)

        return Q


class ActionNetwork(nn.Module):
    def __init__(self):
        super(ActionNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.A = nn.Linear(512, 1)

    def forward(self, observation):
        observation = T.div(observation, 255)
        observation = observation.view(-1, 4, 84, 84)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 64 * 7 * 7)
        observation = F.relu(self.fc1(observation))
        A = self.A(observation)
        return A


class QNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, device):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.Q = nn.Linear(512, n_actions)

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
        observationQ = F.relu(self.fc1(observation))
        Q = self.Q(observationQ)

        return Q

class SepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, device):
        super(SepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.QS = nn.Linear(512, n_actions)
        self.QR = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=0.00015)
        """self.optimizer = optim.SGD(
            [
                {"params": self.conv1.parameters(), lr: 0.00001},
                {"params": self.conv2.parameters(), lr: 0.00001},
                {"params": self.conv3.parameters(), lr: 0.00001}
            ],
            lr=lr)"""

        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def forward(self, observation, sep=False):
        observation = T.div(observation, 255)
        observation = observation.view(-1, 4, 84, 84)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 64 * 7 * 7)
        observationQ = F.relu(self.fc1(observation))
        QR = self.QR(observationQ)
        QS = self.QS(observationQ)

        if sep:
            return QR, QS
        else:
            return QR + QS

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
                 max_mem_size=100000, total_frames=100000, lr=0.0001, discount=0.99,
                 game=None, run=None, name=None):

        self.epsilon = EpsilonGreedy()
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims


        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.min_sampling_size = 1600

        self.chkpt_dir = ""
        self.gamma = discount

        self.run = run
        self.algo_name = name
        self.game = game

        # IMPORTANT params, check these
        self.n = 10
        self.batch_size = 32
        self.duelling = False
        self.aug = False
        self.replace_target_cnt = 1
        self.replay_ratio = 1
        self.network = "normal"
        self.collecting_churn_data = False
        self.gen_data = False
        self.identify_data = False

        self.sep_q_state = False

        self.delayed_reward = False
        self.reward_spread = 5
        self.prev_rewards = deque([0, 0, 0, 0, 0], self.reward_spread)

        if self.identify_data:
            self.identify = Identify(self.min_sampling_size)

        if self.gen_data:
            self.batch_size = 1

        self.memory = NStepExperienceReplay(input_dims, max_mem_size, self.batch_size, self.n, self.gamma)

        if self.network == "normal" and self.duelling:
            self.net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                              input_dims=self.input_dims,
                                              name='lunar_lander_dueling_ddqn_q_eval',
                                              chkpt_dir=self.chkpt_dir, device=device)

            self.tgt_net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                              input_dims=self.input_dims,
                                              name='lunar_lander_dueling_ddqn_q_next',
                                              chkpt_dir=self.chkpt_dir, device=device)

        elif self.network == "normal" and not self.sep_q_state:
            self.net = QNetwork(self.lr, self.n_actions,
                                           input_dims=self.input_dims, device=device)

            self.tgt_net = QNetwork(self.lr, self.n_actions,
                                               input_dims=self.input_dims, device=device)

            if self.replace_target_cnt > 1:
                self.churn_net = QNetwork(self.lr, self.n_actions,
                                        input_dims=self.input_dims, device=device)

        elif self.network == "normal" and self.sep_q_state:
            self.net = SepQNetwork(self.lr, self.n_actions,
                                           input_dims=self.input_dims, device=device)

            self.tgt_net = SepQNetwork(self.lr, self.n_actions,
                                               input_dims=self.input_dims, device=device)

        elif self.network == "split":
            self.net = HydraQNetwork(self.lr, self.n_actions,
                                           input_dims=self.input_dims,
                                           name='lunar_lander_dueling_ddqn_q_eval',
                                           chkpt_dir=self.chkpt_dir, device=device)

            self.tgt_net = HydraQNetwork(self.lr, self.n_actions,
                                               input_dims=self.input_dims,
                                               name='lunar_lander_dueling_ddqn_q_next',
                                               chkpt_dir=self.chkpt_dir, device=device)
        else:
            self.net = LeggedQNetwork(self.lr, self.n_actions,
                                           input_dims=self.input_dims, device=device)

            self.tgt_net = LeggedQNetwork(self.lr, self.n_actions,
                                               input_dims=self.input_dims, device=device)

        #summary(self.net, (4, 84, 84))

        if self.gen_data:
            self.net.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=0.00015)
            self.tgt_net.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=0.00015)

        self.random_shift = nn.Sequential(nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
        self.intensity = Intensity(scale=0.1)

        self.env_steps = 0
        self.grad_steps = 0
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
        self.reward_proportions = True

    def get_grad_steps(self):
        return self.grad_steps

    def set_eval_mode(self):
        self.epsilon.eps_final = 0.05
        self.epsilon.eps = 0.05

    def choose_action(self, observation):
        if self.identify_data and self.memory.mem_cntr >= self.min_sampling_size:
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
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.net.device)
            q_vals = self.net.forward(state)
            action = T.argmax(q_vals).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        if self.delayed_reward:
            self.prev_rewards.appendleft(reward)
            if done:
                reward = 0
                for i in range(len(self.prev_rewards)):
                    reward += self.prev_rewards[i] * ((self.reward_spread - i) / self.reward_spread)

                self.prev_rewards = deque([0, 0, 0, 0, 0], self.reward_spread)
            else:
                reward = 0
                for i in self.prev_rewards:
                    reward += i / self.reward_spread

        self.memory.store_transition(state, action, reward, state_, done)
        self.env_steps += 1
        self.total_actions[action] += 1

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
        for i in range(self.replay_ratio):
            self.learn_call()

    def learn_call(self):

        if self.memory.mem_cntr < self.min_sampling_size:
            return

        self.net.optimizer.zero_grad()

        if self.replace_target_cnt > 1:
            self.churn_net.load_state_dict(self.net.state_dict())

        if self.grad_steps % self.replace_target_cnt == 0:
            self.replace_target_network()

        states, actions, rewards, new_states, dones = self.memory.sample_memory()

        states = T.tensor(states).to(self.net.device)
        rewards = T.tensor(rewards).to(self.net.device)
        dones = T.tensor(dones).to(self.net.device)
        actions = T.tensor(actions).to(self.net.device)
        states_ = T.tensor(new_states).to(self.net.device)

        indices = np.arange(self.batch_size)

        if self.aug:
            states_aug = (self.intensity(self.random_shift(states.float()))).to(T.uint8)
            states_aug_ = (self.intensity(self.random_shift(states_.float()))).to(T.uint8)
            states_aug_policy_ = (self.intensity(self.random_shift(states_.float()))).to(T.uint8)
            q_pred = self.net.forward(states_aug)  # states_aug
            q_targets = self.tgt_net.forward(states_aug_)  # states_aug_
            q_actions = self.net.forward(states_aug_policy_)  # states_aug_policy_
        elif self.sep_q_state:
            qr_pred, qs_pred = self.net.forward(states, sep=True)  # states_aug
            q_targets = self.tgt_net.forward(states_)  # states_aug_
            q_actions = self.net.forward(states_)

            qr_pred = qr_pred[indices, actions]
            qs_pred = qs_pred[indices, actions]

            with torch.no_grad():
                max_actions = T.argmax(q_actions, dim=1)
                q_targets[dones] = 0.0

                q_next_state_targets = (self.gamma ** self.n) * q_targets[indices, max_actions]

            reward_loss = (qr_pred - rewards) ** 2
            next_state_loss = (qs_pred - q_next_state_targets) ** 2
            loss = reward_loss + next_state_loss
            loss = loss.mean().to(self.net.device)

        else:
            q_pred = self.net.forward(states)  # states_aug
            q_targets = self.tgt_net.forward(states_)  # states_aug_
            q_actions = self.net.forward(states_)

        if self.identify_data:
            q_pred_og = q_pred.detach().cpu()

        if not self.sep_q_state:
            q_pred = q_pred[indices, actions]

            with torch.no_grad():
                max_actions = T.argmax(q_actions, dim=1)
                q_targets[dones] = 0.0

                q_target = rewards + (self.gamma ** self.n) * q_targets[indices, max_actions]

                if self.reward_proportions:
                    self.reward_target_avg.append(abs(float(rewards.mean().cpu())))
                    self.bootstrap_target_avg.append(abs(float(((self.gamma ** self.n) * q_targets[indices, max_actions]).mean().cpu())))

            loss = self.net.loss(q_target, q_pred).to(self.net.device)

        loss.backward()
        T.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.net.optimizer.step()

        self.grad_steps += 1

        self.epsilon.update_eps()

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


