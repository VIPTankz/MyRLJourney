import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer(object):
    def __init__(self, obs_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, 1), dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def fetch(self, idxs, discount, n):
        assert idxs.max() + n <= len(self)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs + n - 1]

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = np.zeros((idxs.shape[0], 1), dtype=np.float32)
        not_dones = np.ones((idxs.shape[0], 1), dtype=np.float32)
        for i in range(n):
            rewards += (discount**n) * not_dones * np.sign(
                self.rewards[idxs + i])
            not_dones = np.minimum(not_dones, self.not_dones[idxs + i])

        rewards = torch.as_tensor(rewards, device=self.device)
        not_dones = torch.as_tensor(not_dones, device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def sample_idxs(self, batch_size, n):
        last_idx = (self.capacity if self.full else self.idx) - (n - 1)
        idxs = np.random.randint(0, last_idx, size=batch_size)

        return idxs

    def sample_multistep(self, batch_size, discount, n):
        assert n <= self.idx or self.full
        idxs = self.sample_idxs(batch_size, n)

        return self.fetch(idxs, discount, n)
