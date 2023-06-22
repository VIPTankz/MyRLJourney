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
from PrioritisedExperienceReplaySPR import PrioritizedReplayBuffer
import kornia.augmentation as aug
from ema_pytorch import EMA


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

    def forward(self, inp):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: T.sign(x) * T.sqrt(T.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = T.mul(eps_in, eps_out)
        return F.linear(inp, self.weight + self.sigma_weight * noise_v, bias)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """
        takes a framestack of images or batch of framestack of images
        Note: This returns unflattened output
        """

        x = x.float() / 256
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x


class MLPLayer1(nn.Module):
    def __init__(self, input_size):
        """
        This is just a single layer. Is used by both the q-learning head and the projection

        For atari input size will be 64*7*7
        """
        super(MLPLayer1, self).__init__()
        self.fc1V = nn.Linear(input_size, 256)
        self.fc1A = nn.Linear(input_size, 256)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """
        takes flattened output from the decoder
        can also take output from the transition model
        """
        V = F.relu(self.fc1V(x))
        A = F.relu(self.fc1A(x))

        return V, A


class QLearningHeadFinal(nn.Module):
    def __init__(self, n_actions, atoms, Vmax, Vmin):
        """
        This is only the final layer of the Q-Learning Head
        """
        super(QLearningHeadFinal, self).__init__()

        self.V = NoisyFactorizedLinear(256, atoms)
        self.A = NoisyFactorizedLinear(256, n_actions * atoms)

        self.atoms = atoms
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.DELTA_Z = (self.Vmax - self.Vmin) / (self.atoms - 1)

        self.register_buffer("supports", T.arange(Vmin, Vmax + self.DELTA_Z, self.DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, V, A):
        """
        Takes input from the first MLP Layer
        """
        batch_size = A.size()[0]
        adv_out = self.A(A).view(batch_size, -1, self.atoms)
        val_out = self.V(V).view(batch_size, 1, self.atoms)

        adv_mean = adv_out.mean(dim=1, keepdim=True)
        cat_out = val_out + (adv_out - adv_mean)

        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        qvals = weights.sum(dim=2)
        return cat_out, qvals

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, self.atoms)).view(t.size())


class QHead(nn.Module):
    def __init__(self):
        """
        This is the Q head for after the online projection
        NOT to be confused with the q-learning head, they are different
        """
        super(QHead, self).__init__()

        self.q_head = nn.Linear(512, 512)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """
        Takes input from the output of the first MLP layer, which in turn comes from the transition model
        This produces the final online latent, or y_hat_(t+k) in mathy terms
        """
        return self.q_head(x)


class TransitionModel(nn.Module):
    def __init__(self, n_actions):
        super(TransitionModel, self).__init__()

        self.conv_trans1 = nn.Conv2d(64 + n_actions, 64, 3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv_trans2 = nn.Conv2d(64, 64, 3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.n_actions = n_actions

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, actions):
        """
        Takes the output of the online encoder, and produces Z_hat_(t+k)
        This is also what is bootstrapped/iterated over
        ie the output of this network can go back into itself
        Output is not flattened for this reason
        """
        batch_size = x.size()[0]

        # adding onehot vector
        batch_range = T.arange(batch_size, device=self.device)
        action_onehot = T.zeros(batch_size, self.n_actions, x.shape[-2], x.shape[-1], device=self.device)
        action_onehot[batch_range, actions, :, :] = 1
        x = T.cat([x, action_onehot], 1)

        x = self.batch_norm1(F.relu(self.conv_trans1(x)))
        z_hat = self.batch_norm2(F.relu(self.conv_trans2(x)))

        return z_hat


class SPRNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, chkpt_dir, atoms, Vmax, Vmin, K, batch_size):
        super(SPRNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.K = K
        self.learn_batch_size = batch_size

        self.encoder = Encoder()
        self.tgt_encoder = EMA(self.encoder, beta=0.0, update_after_step=0, update_every=1)
        self.conv_output_size = 64 * 7 * 7

        self.mlp_layer1 = MLPLayer1(self.conv_output_size)
        self.tgt_mlp_layer1 = EMA(self.mlp_layer1, beta=0.0, update_after_step=0, update_every=1)

        self.qlearning_head = QLearningHeadFinal(n_actions, atoms, Vmax, Vmin)

        self.q_head = QHead()

        self.transition_model = TransitionModel(n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def update_EMAs(self):
        self.tgt_encoder.update()
        self.tgt_mlp_layer1.update()

    def encode(self, x):
        """
        This returns unflattened output of conv layers
        """
        return self.encoder(x)

    def tgt_encode(self, x):
        """
        same as encode with tgt network
        """
        with T.no_grad():
            x = self.tgt_encoder(x)

        return x

    def decode(self, x):
        """
        Takes input from encoder, and produces both the categorical outputs and qvals
        """
        batch_size = x.size()[0]

        # flatten conv layer output
        conv_out = x.view(batch_size, -1)
        V, A = self.mlp_layer1(conv_out)
        cat_out, qvals = self.qlearning_head(V, A)
        return cat_out, qvals

    def forward(self, x):
        """
        This method should only be used for action selection
        """
        x = self.encode(x)
        _, qvals = self.decode(x)
        return qvals

    def produce_online_latents(self, x, actions):
        """
        Input x should be output from encoder

        actions should come from replay buffer.
        Remember actions should be one step behind next_state!

        returns tensor of latents, of length K
        """
        batch_size = x.size()[0]
        latents = []
        for i in range(self.K):
            # this gives what needs to be fed back into network
            x = self.transition_model(x, actions[:, i])

            flat_x = x.view(batch_size, -1)

            proj_v, proj_a = self.mlp_layer1(flat_x)

            # concat v and a
            proj = T.cat([proj_v, proj_a], dim=1)

            latent = self.q_head(proj)
            latents.append(latent)

        latents = T.stack(latents).to(self.device)
        latents = T.swapaxes(latents, 0, 1)
        return latents

    def calculate_tgt_latents(self, x):
        """
        x should be a tensor of the next k obsevations (framestacks)

        the input size will actually be [160x4x84x84]
        this is because 5*32 = 160, and we need to do them all anyway
        this is fine but they need to be reshaped after

        returns tensor of target latents. In mathy terms, y_tilda_(t+k)
        """
        with T.no_grad():
            batch_size = x.size()[0]
            conv_out = self.tgt_encode(x)

            # flatten conv layer output
            conv_out = conv_out.view(batch_size, -1)

            latents_v, latents_a = self.tgt_mlp_layer1(conv_out)
            latents = T.cat([latents_v, latents_a], dim=1)

            # reshape back to [batch_size x K x 512]
            latents = T.reshape(latents, (self.learn_batch_size, self.K, -1))

        return latents

    def spr_loss(self, f_x1s, f_x2s):
        """
        Need to check this can be used in a batch
        probably check the shape and things
        """
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        # Gradients of normalized L2 loss and cosine similarity are proportional.
        # See: https://stats.stackexchange.com/a/146279
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
        return loss

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims,
                 max_mem_size=100000, replace=1, total_frames=100000, lr=0.0001, batch_size=32, discount=0.99):

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
        self.spr_loss_coef = 2
        self.K = 5

        # n-step
        self.n = 10
        self.nstep_states = deque([], self.n)
        self.nstep_rewards = deque([], self.n)
        self.nstep_actions = deque([], self.n)

        # c51
        self.Vmax = 10
        self.Vmin = -10
        self.N_ATOMS = 51

        self.memory = PrioritizedReplayBuffer(input_dims, n_actions, max_mem_size, eps=1e-5, alpha=0.5, beta=0.4,
                                              total_frames=total_frames)

        self.net = SPRNetwork(self.lr, self.n_actions, name='lunar_lander_dueling_ddqn_q_eval',
                              chkpt_dir=self.chkpt_dir, atoms=self.N_ATOMS, Vmax=self.Vmax, Vmin=self.Vmin, K=self.K,
                              batch_size=self.batch_size)

        # augmentation
        self.random_shift = nn.Sequential(nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
        self.intensity = Intensity(scale=0.05)

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.net.device)
        advantage = self.net(state)
        return T.argmax(advantage).item()

    def get_grad_steps(self):
        return self.grad_steps

    def store_transition(self, state, action, reward, state_, done):
        self.n_step(state, action, reward, state_, done)

    def n_step(self, state, action, reward, state_, done):
        self.nstep_states.append(state)
        self.nstep_rewards.append(reward)
        self.nstep_actions.append(action)

        if len(self.nstep_states) == self.n:
            fin_reward = 0
            for i in range(self.n):
                fin_reward += self.nstep_rewards[i] * (self.gamma ** i)
            self.memory.add(self.nstep_states[0], self.nstep_actions[0], fin_reward, state_, done)

        if done:
            self.nstep_states = deque([], self.n)
            self.nstep_rewards = deque([], self.n)
            self.nstep_actions = deque([], self.n)

    def save_models(self):
        self.net.save_checkpoint()

    def load_models(self):
        self.net.load_checkpoint()

    def learn(self):

        if self.memory.count < self.min_sampling_size:
            return

        self.net.optimizer.zero_grad()

        # update EMAs
        self.net.update_EMAs()

        batch, weights, tree_idxs = self.memory.sample(self.batch_size)

        states, actions, rewards, new_states, dones, future_states, future_actions, future_dones = batch

        states = states.to(self.net.device)
        rewards = rewards.to(self.net.device)
        dones = dones.to(self.net.device)
        actions = actions.to(self.net.device)
        states_ = new_states.to(self.net.device)
        weights = weights.to(self.net.device)
        future_states = future_states.to(self.net.device)
        future_actions = future_actions.to(self.net.device)
        future_dones = future_dones.to(self.net.device)

        # Data Augmentation

        states = (self.intensity(self.random_shift(states.float() / 255.)) * 255).to(T.uint8)
        states_ = (self.intensity(self.random_shift(states_.float() / 255.)) * 255).to(T.uint8)

        # Forward passes on network
        states_next_states = T.cat((states, states_))
        encodings = self.net.encode(states_next_states)

        distr_v, qvals_v = self.net.decode(encodings)

        next_qvals_v = qvals_v[self.batch_size:]
        next_distr_v = distr_v[self.batch_size:]
        distr_v = distr_v[:self.batch_size]

        # Rainbow Loss Code
        next_actions_v = next_qvals_v.max(1)[1]

        next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
        next_best_distr_v = self.net.qlearning_head.apply_softmax(next_best_distr_v)
        next_best_distr = next_best_distr_v.data.cpu()

        proj_distr = distr_projection(next_best_distr, rewards.cpu(), dones.cpu(), self.Vmin, self.Vmax, self.N_ATOMS,
                                      self.gamma)

        state_action_values = distr_v[range(self.batch_size), actions.data]
        state_log_sm_v = F.log_softmax(state_action_values, dim=1)
        proj_distr_v = proj_distr.to(self.net.device)

        loss_v = -state_log_sm_v * proj_distr_v
        weights = T.squeeze(weights)
        loss_v = weights.to(self.net.device) * loss_v.sum(dim=1)

        rainbow_loss = loss_v.mean()
        # SPR Loss Code
        pred_latents = self.net.produce_online_latents(encodings[:self.batch_size], future_actions)
        target_latents = self.net.calculate_tgt_latents(future_states)

        # Remove done trajectories
        for i in range(self.K - 1):
            future_dones[:, i + 1] = T.logical_or(future_dones[:, i], future_dones[:, i + 1])

        # this just sets elements equal if they are during or after done, making them give 0 loss
        future_dones = future_dones.to(T.bool)
        pred_latents[future_dones] = target_latents[future_dones]

        spr_loss = self.net.spr_loss(pred_latents, target_latents)
        spr_loss = spr_loss.sum()

        # Final Loss
        loss = rainbow_loss + self.spr_loss_coef * spr_loss
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.net.optimizer.step()
        self.learn_step_counter += 1

        self.memory.update_priorities(tree_idxs, loss_v.cpu().detach().numpy())


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Categorical Algorithm from the
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
