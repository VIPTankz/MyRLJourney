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

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: T.sign(x) * T.sqrt(T.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = T.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)

class TransitionModel(nn.Module):
    def __init__(self,n_actions,device):
        super(TransitionModel,self).__init__()
        self.conv_trans1 = nn.Conv2d(64 + n_actions, 64, 3,padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv_trans2 = nn.Conv2d(64, 64, 3,padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.device = device
        self.n_actions = n_actions

        self.to(self.device)
    def forward(self,x ,action):
        #takes the output of conv
        batch_size = x.size()[0]
        x = T.reshape(x,(32,64,7,7))

        #adding onehot vector
        batch_range = T.arange(action.shape[0], device=self.device)
        action_onehot = T.zeros(batch_size,self.n_actions,x.shape[-2],x.shape[-1],device=self.device)
        action_onehot[batch_range, action, :, :] = 1
        x = T.cat([x, action_onehot], 1)

        x = self.batch_norm1(F.relu(self.conv_trans1(x)))

        next_latent = self.batch_norm2(F.relu(self.conv_trans2(x)))

        next_latent = next_latent.view(batch_size,-1)

        return next_latent

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir,atoms,Vmax,Vmin,K):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.atoms = atoms
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.DELTA_Z = (self.Vmax - self.Vmin) / (self.atoms - 1)
        self.n_actions = n_actions
        self.K = K

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print("Device: " + str(self.device),flush=True)

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.tgt_conv1 = EMA(self.conv1,beta=0.0,update_after_step=0,update_every=1)
        self.tgt_conv2 = EMA(self.conv2, beta=0.0, update_after_step=0, update_every=1)
        self.tgt_conv3 = EMA(self.conv3, beta=0.0, update_after_step=0, update_every=1)

        self.trans_net = TransitionModel(n_actions,self.device)

        self.fc1V = nn.Linear(64 * 7 * 7, 256)
        self.tgt_fc1V = EMA(self.fc1V,beta=0.0,update_after_step=0,update_every=1)
        self.fc1A = nn.Linear(64 * 7 * 7, 256)
        self.tgt_fc1A = EMA(self.fc1A,beta=0.0,update_after_step=0,update_every=1)
        self.V = NoisyFactorizedLinear(256, atoms)
        self.A = NoisyFactorizedLinear(256, n_actions * atoms)

        self.proj_q_head = nn.Linear(512, 512)

        self.register_buffer("supports", T.arange(Vmin, Vmax + self.DELTA_Z, self.DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.to(self.device)

    def get_spr_latents(self,conv_out):
        #x should be the output of conv

        latents = []
        batch_size = conv_out.size(dim=0)
        for i in range(self.K):
            V = self.fc_val(conv_out).view(batch_size, 1, self.atoms)
            A = self.fc_adv(conv_out).view(batch_size, -1, self.atoms)
            adv_mean = A.mean(dim=1, keepdim=True)
            cat_out = V + (A - adv_mean)
            probs = self.apply_softmax(cat_out)
            weights = probs * self.supports
            qvals = weights.sum(dim=2)
            action = T.argmax(qvals,dim=1)

            conv_out = self.trans_net(conv_out,action)
            flat_conv_trans_out = conv_out.view(batch_size, -1)

            latent = self.projection(flat_conv_trans_out)

            latents.append(T.clone(latent))

        return latents

    def projection(self,flat_conv_out,tgt=False):
        if not tgt:
            latent_v = F.relu(self.fc1V(flat_conv_out))
            latent_a = F.relu(self.fc1A(flat_conv_out))
            latent = T.cat([latent_v, latent_a], dim=1)
            latent = self.proj_q_head(latent)
            return latent

        else:
            latent_v = F.relu(self.tgt_fc1V(flat_conv_out))
            latent_a = F.relu(self.tgt_fc1A(flat_conv_out))
            latent = T.cat([latent_v, latent_a], dim=1)
            return latent

    def get_spr_loss(self,pred_latents,obs_latents,future_dones):
        tot_loss = 0

        batch_size = pred_latents[0].size(dim=0)
        mask = T.zeros(batch_size,dtype=T.bool)

        for i in range(self.K):
            c_mask = T.Tensor(future_dones[i])
            mask = T.logical_or(mask,c_mask)

            # this just sets elements equal if they are during or after done, making them give 0 loss
            pred_latents[i][mask] = obs_latents[i][mask]

            tot_loss += self.spr_loss(pred_latents[i],obs_latents[i])

        print(tot_loss)
        return tot_loss

    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        # Gradients of norrmalized L2 loss and cosine similiarity are proportional.
        # See: https://stats.stackexchange.com/a/146279
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
        return loss

    def conv(self,x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x

    def tgt_conv(self,x):

        x = F.relu(self.tgt_conv1(x))
        x = F.relu(self.tgt_conv2(x))
        x = F.relu(self.tgt_conv3(x))

        return x

    def fc_val(self,x):
        x = F.relu(self.fc1V(x))
        x = self.V(x)

        return x

    def fc_adv(self,x):
        x = F.relu(self.fc1A(x))
        x = self.A(x)

        return x

    def get_latent_from_obs(self,x,tgt=False):
        #gets latents from images
        batch_size = x.size()[0]
        fx = x.float() / 256
        if not tgt:
            return self.conv(fx).view(batch_size, -1)
        else:
            return self.tgt_conv(fx).view(batch_size, -1)

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1, self.atoms)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, self.atoms)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

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
    def __init__(self, n_actions,input_dims,
                 max_mem_size=100000, total_frames=100000,lr=0.0001,batch_size=32,discount=0.99):

        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.min_sampling_size = 2000
        self.chkpt_dir = ""
        self.gamma = discount
        self.eval_mode = False
        self.grad_steps = 2
        self.K = 5

        #n-step
        self.n = 10
        self.nstep_states = deque([], self.n)
        self.nstep_rewards = deque([], self.n)
        self.nstep_actions = deque([], self.n)

        #c51
        self.Vmax = 10
        self.Vmin = -10
        self.N_ATOMS = 51

        self.spr_loss_coef = 2

        self.memory = PrioritizedReplayBuffer(input_dims, n_actions, max_mem_size, eps=1e-5, alpha=0.5, beta=0.4,
                                                  total_frames=total_frames)

        self.net = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_eval',
                                          chkpt_dir=self.chkpt_dir,
                                          atoms=self.N_ATOMS,Vmax=self.Vmax,Vmin=self.Vmin,
                                          K=self.K)

        self.random_shift = nn.Sequential(nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
        self.intensity = Intensity(scale=0.05)

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.net.device)
        advantage = self.net.qvals(state)
        return T.argmax(advantage).item()


    def store_transition(self, state, action, reward, state_, done):
        self.n_step(state, action, reward, state_, done)

    def get_grad_steps(self):
        return self.grad_steps

    def n_step(self, state, action, reward, state_, done):
        self.nstep_states.append(state)
        self.nstep_rewards.append(reward)
        self.nstep_actions.append(action)

        if len(self.nstep_states) == self.n:
            fin_reward = 0
            for i in range(self.n):
                fin_reward += self.nstep_rewards[i] * (self.gamma ** i)
            self.memory.add(self.nstep_states[0], self.nstep_actions[0],fin_reward, \
                                         state_, done)

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

        self.net.tgt_conv1.update()
        self.net.tgt_conv2.update()
        self.net.tgt_conv3.update()
        self.net.tgt_fc1A.update()
        self.net.tgt_fc1V.update()

        batch, weights, tree_idxs = self.memory.sample(self.batch_size)

        #need to get the future idxs and calculate the true latents

        #BEWARE: this has a known error where we can sample from outside of current space

        states, actions, rewards, new_states, dones, future_states, future_dones = batch

        states = states.to(self.net.device)
        rewards = rewards.to(self.net.device)
        dones = dones.to(self.net.device)
        actions = actions.to(self.net.device)
        states_ = new_states.to(self.net.device)
        weights = weights.to(self.net.device)

        states_aug = (self.intensity(self.random_shift(states.float()/255.)) * 255).to(T.uint8)
        states_aug_ = (self.intensity(self.random_shift(states_.float()/255.)) * 255).to(T.uint8)

        for i in range(self.K):
            future_states[i] = future_states[i].to(self.net.device)

        for i in range(self.K):
            future_states[i] = (self.intensity(self.random_shift(future_states[i].float()/255.)) * 255).to(T.uint8)

        obs_latents = []
        for i in range(self.K):
            obs_latents.append(self.net.get_latent_from_obs(future_states[i],tgt=True))

        for i in range(len(obs_latents)):
            obs_latents[i] = self.net.projection(obs_latents[i],tgt=True)

        ##############
        distr_v, qvals_v = self.net.both(T.cat((states_aug, states_aug_)))
        next_qvals_v = qvals_v[self.batch_size:]
        distr_v = distr_v[:self.batch_size]

        next_actions_v = next_qvals_v.max(1)[1]
        next_distr_v = self.net(states_aug_)
        next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
        next_best_distr_v = self.net.apply_softmax(next_best_distr_v)
        next_best_distr = next_best_distr_v.data.cpu()

        proj_distr = distr_projection(next_best_distr, rewards.cpu(), dones.cpu(), self.Vmin, self.Vmax, self.N_ATOMS, self.gamma)

        state_action_values = distr_v[range(self.batch_size), actions.data]
        state_log_sm_v = F.log_softmax(state_action_values, dim=1)
        proj_distr_v = proj_distr.to(self.net.device)

        loss_v = -state_log_sm_v * proj_distr_v
        weights = T.squeeze(weights)
        loss_v = weights.to(self.net.device) * loss_v.sum(dim=1)

        rainbow_loss = loss_v.mean()

        latents = self.net.get_latent_from_obs(states_aug)
        pred_latents = self.net.get_spr_latents(latents)

        spr_loss = self.net.get_spr_loss(pred_latents,obs_latents,future_dones)

        loss = rainbow_loss + self.spr_loss_coef * spr_loss
        print("hi")
        ##############

        loss.backward()
        T.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.net.optimizer.step()
        self.learn_step_counter += 1

        self.memory.update_priorities(tree_idxs, loss_v.cpu().detach().numpy())

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


