import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ActionNetwork(nn.Module):
    def __init__(self):
        super(ActionNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.A = nn.Linear(512, 1)

    def forward(self, observation):
        observation = torch.div(observation, 255)
        observation = observation.view(-1, 4, 84, 84)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 64 * 7 * 7)
        observation = F.relu(self.fc1(observation))
        A = self.A(observation)
        return A


class LeggedQNetwork(nn.Module):
    def __init__(self, lr, n_actions, device):
        super(LeggedQNetwork, self).__init__()
        self.n_actions = n_actions

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

        A = torch.stack(outs, dim=1)

        return A

    def choose_action(self, observation):
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(self.device)
        advantage = self.forward(state)
        action = torch.argmax(advantage).item()

        return action

if __name__ == "__main__":

    bs = 1
    n_actions = 4

    discount = 0.99
    n = 3
    device = 'cuda:0'

    net = LeggedQNetwork(0.0001, n_actions, device)
    tgt_net = LeggedQNetwork(0.0001, n_actions, device)

    indices = np.arange(bs)

    for i in range(10):

        x = np.random.randint(low=0, high=256, size=(4, 84, 84,))
        action_null = net.choose_action(x)

        tgt_net.load_state_dict(net.state_dict())

        net.optimizer.zero_grad()

        rewards = torch.randn((bs,)).to(net.device)
        x = torch.randint(low=0, high=256, size=(bs, 4, 84, 84,)).to(net.device)
        x_ = torch.randint(low=0, high=256, size=(bs, 4, 84, 84,)).to(net.device)
        actions = torch.randint(low=0, high=n_actions, size=(bs,))
        dones = torch.cuda.FloatTensor(bs).uniform_() > 0.9

        xs = net(x)
        x_targets = tgt_net(x_)
        x_action = net(x_)

        q_pred = xs
        q_next = x_targets
        q_eval = x_action

        x_0 = q_pred[indices, actions]

        max_actions = torch.argmax(q_eval, dim=1)
        q_next[dones] = 0

        target = rewards + (discount ** n) * q_next[indices, max_actions]

        loss = net.loss(x_0, target).to(net.device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)

        net.optimizer.step()

        with torch.no_grad():

            print("\nBefore")
            print(xs)
            print("After")
            print(net(x))
            print("Difference")
            print(xs - net(x))

