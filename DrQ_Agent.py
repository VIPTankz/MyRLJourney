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
                 max_mem_size=100000, replace=1,total_frames=100000,lr=0.0001,batch_size=32,discount=0.99,
                 game=None, run=None):

        self.epsilon = EpsilonGreedy()
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = 32
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.min_sampling_size = 1600
        self.n = 10
        self.chkpt_dir = ""
        self.gamma = discount
        self.grad_steps = 8
        self.run = run
        self.algo_name = "EffDQN"

        self.collecting_churn_data = False
        self.gen_data = False

        self.memory = NStepExperienceReplay(input_dims, max_mem_size, self.batch_size, self.n, self.gamma)

        self.net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_eval',
                                          chkpt_dir=self.chkpt_dir, device=device)

        self.tgt_net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_next',
                                          chkpt_dir=self.chkpt_dir, device=device)

        self.random_shift = nn.Sequential(nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
        self.intensity = Intensity(scale=0.1)

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

        self.action_changes = np.zeros(self.n_actions, dtype=np.int64)
        self.action_changes2 = np.zeros(self.n_actions, dtype=np.int64)
        self.action_changes3 = np.zeros(self.n_actions, dtype=np.int64)
        self.steps = 10000
        self.lists = []
        self.lists2 = []
        self.lists3 = []

    def get_grad_steps(self):
        return self.grad_steps

    def set_eval_mode(self):
        self.epsilon.eps_final = 0.05
        self.epsilon.eps = 0.05

    def choose_action(self, observation):
        if np.random.random() > self.epsilon.eps:
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.net.device)
            _, advantage = self.net.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
        self.env_steps += 1
        self.total_actions[action] += 1

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

        if self.memory.mem_cntr < self.min_sampling_size:
            return

        self.net.optimizer.zero_grad()

        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.replace_target_network()

        states, actions, rewards, new_states, dones = self.memory.sample_memory()

        states = T.tensor(states).to(self.net.device)
        rewards = T.tensor(rewards).to(self.net.device)
        dones = T.tensor(dones).to(self.net.device)
        actions = T.tensor(actions).to(self.net.device)
        states_ = T.tensor(new_states).to(self.net.device)

        indices = np.arange(self.batch_size)

        states_aug = (self.intensity(self.random_shift(states.float()))).to(T.uint8)
        states_aug_ = (self.intensity(self.random_shift(states_.float()))).to(T.uint8)
        states_aug_policy_ = (self.intensity(self.random_shift(states_.float()))).to(T.uint8)

        V_s, A_s = self.net.forward(states_aug)  # states_aug

        V_s_, A_s_ = self.tgt_net.forward(states_aug_)  # states_aug_

        V_s_eval, A_s_eval = self.tgt_net.forward(states_aug_policy_)  # states_aug_policy_

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

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

        if self.gen_data:

            with T.no_grad():
                V_s_af, A_s_af = self.net.forward(states)  # states_aug

                A_s_af_copy = T.clone(A_s_af)
                A_s_copy = T.clone(A_s)
                for i in range(len(A_s_af_copy[0])):

                    if i != actions[0]:
                        A_s_af_copy[0, i] = 0
                        A_s_copy[0, i] = 0

                self.action_changes = np.add(self.action_changes, (T.abs(A_s_af - A_s)).detach().cpu().numpy())
                A_s_af[0, actions[0]] = 0
                A_s[0, actions[0]] = 0
                self.action_changes2 = np.add(self.action_changes2, (T.abs(A_s_af - A_s)).detach().cpu().numpy())

                self.action_changes3 = np.add(self.action_changes3, (T.abs(A_s_af_copy - A_s_copy)).detach().cpu().numpy())
                #print(self.action_changes)
                #print(self.action_changes.sum())
                #print("\n\n")
                self.lists.append([self.action_changes[0,0],self.action_changes[0,1],self.action_changes[0,2],self.action_changes[0,3],self.action_changes.sum()])
                self.lists2.append([self.action_changes2[0, 0], self.action_changes2[0, 1], self.action_changes2[0, 2],
                                   self.action_changes2[0, 3], self.action_changes2.sum()])
                self.lists3.append([self.action_changes3[0, 0], self.action_changes3[0, 1], self.action_changes3[0, 2],
                                   self.action_changes3[0, 3], self.action_changes3.sum()])

                if self.learn_step_counter % self.steps == 0:
                    y = np.arange(len(self.lists))
                    count = 0
                    for i in range(5):
                        if count != 4:
                            label = "Action " + str(count)
                        else:
                            label = "Total"
                        plt.plot(y, np.array(self.lists)[:,count], label=label)  # Plot the chart
                        count += 1
                    plt.legend()
                    plt.title("Total Change")
                    plt.show()  # display
                    plt.clf()
                    y = np.arange(len(self.lists))
                    count = 0
                    for i in range(5):
                        if count != 4:
                            label = "Action " + str(count)
                        else:
                            label = "Total"
                        plt.plot(y, np.array(self.lists2)[:,count], label=label)  # Plot the chart
                        count += 1
                    plt.legend()
                    plt.title("Change Via Generalisation")
                    plt.show()  # display

                    plt.clf()
                    y = np.arange(len(self.lists))
                    count = 0
                    for i in range(5):
                        if count != 4:
                            label = "Action " + str(count)
                        else:
                            label = "Total"
                        plt.plot(y, np.array(self.lists3)[:,count], label=label)  # Plot the chart
                        count += 1
                    plt.legend()
                    plt.title("Change Via On Targetted Action")
                    plt.show()  # display

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

        _, cur_vals = self.net(states)
        _, tgt_vals = self.tgt_net(states)

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


