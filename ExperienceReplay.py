import numpy as np

class ExperienceReplay:
    def __init__(self, input_dims, max_mem, batch_size):
        self.mem_size = max_mem
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.uint8)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.uint8)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def __len__(self):
        return self.mem_cntr

    def sample_memory(self, bs=None):
        if bs is None:
            bs = self.batch_size

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, bs, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminals

    def is_sufficient(self):
        return self.mem_cntr > self.batch_size

class NStepExperienceReplay:
    def __init__(self, input_dims, max_mem, batch_size, n, discount):
        self.mem_size = max_mem
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.mem_cntr = 0
        self.n = n
        self.discount = discount
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.uint8)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.uint8)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def __len__(self):
        return self.mem_cntr

    def sample_memory(self, bs=None):
        if bs is None:
            bs = self.batch_size

        if self.mem_cntr >= self.mem_size:
            max_mem = self.mem_size
            batch = np.random.choice(max_mem, bs, replace=False)
            illegals = np.arange(self.n - 1) + self.mem_cntr - self.n + 1
            while np.any(np.in1d(batch, illegals)):
                idxs = np.in1d(batch, illegals)
                subbatch = np.random.choice(max_mem, bs, replace=False)
                batch[idxs] = subbatch[idxs]

        else:
            max_mem = self.mem_cntr - self.n
            batch = np.random.choice(max_mem, bs, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]

        rewards = np.zeros(bs, dtype=np.float32)

        terminals = np.zeros(bs, dtype=np.bool_)

        for i in range(self.n):

            reward_batch = self.reward_memory[(batch + i) % self.mem_size]
            reward_batch[terminals] = 0
            rewards += reward_batch * (self.discount ** i)

            terminals = np.logical_or(terminals, self.terminal_memory[(batch + i) % self.mem_size])

        # if there is a terminal, this will be zeroed out anyway
        new_states = self.new_state_memory[(batch + self.n - 1) % self.mem_size]

        return states, actions, rewards, new_states, terminals

    def is_sufficient(self):
        return self.mem_cntr > self.batch_size

if __name__ == "__main__":
    mem = NStepExperienceReplay([1], 10, 3, 3, 0.5)

    mem.store_transition(1, 2, 1, 2, False)
    mem.store_transition(2, 2, 2, 3, False)
    mem.store_transition(3, 2, 3, 4, False)
    mem.store_transition(4, 2, 4, 5, True)
    mem.store_transition(11, 2, -1, 12, False)
    mem.store_transition(12, 2, -2, 13, False)
    mem.store_transition(13, 2, -3, 14, False)
    mem.store_transition(14, 2, -3, 15, False)
    mem.store_transition(15, 2, -3, 16, False)
    mem.store_transition(16, 2, -3, 17, False)
    mem.store_transition(17, 2, -3, 18, False)

    while True:
        states, actions, rewards, new_states, dones = mem.sample_memory(3)

        print("States: " + str(states))
        print("actions: " + str(actions))
        print("rewards: " + str(rewards))
        print("new_states: " + str(new_states))
        print("dones: " + str(dones))
