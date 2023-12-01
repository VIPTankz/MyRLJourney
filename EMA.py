import torch
from torch import nn
from copy import deepcopy


class EMA(nn.Module):
    def __init__(self, net, tau):
        super(EMA, self).__init__()

        self.net = net
        self.tau = tau  # smoothing factor

        self.ema_net = deepcopy(self.net)

    def forward(self, x, use_noise=None):
        with torch.no_grad():
            if use_noise is None:
                x = self.ema_net(x)
            else:
                x = self.ema_net(x, use_noise)

        return x

    def decode(self, x):
        with torch.no_grad():
            x = self.ema_net.decode(x)

        return x

    def encode(self, x):
        with torch.no_grad():
            x = self.ema_net.encode(x)

        return x

    def both(self, x):
        with torch.no_grad():
            x = self.net.both(x)

        return x

    def apply_softmax(self, x):
        with torch.no_grad():
            x = self.net.softmax(x)

        return x

    def update(self):

        if self.tau == 1:
            state_dict = self.net.state_dict()
            self.ema_net.load_state_dict(state_dict)
        elif self.tau > 0:
            for param, target_param in zip(self.net.parameters(), self.ema_net.parameters()):
                target_param.data.copy_(self.tau * param.data +
                                        (1 - self.tau) * target_param.data)

    def load_state_dict(self, state_dict):
        self.ema_net.load_state_dict(state_dict)


if __name__ == "__main__":
    pass
