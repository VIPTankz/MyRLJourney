import torch
from torch import nn
from copy import deepcopy


class EMA(nn.Module):
    def __init__(self, net, tau):
        super(EMA, self).__init__()

        self.net = net
        self.tau = tau  # smoothing factor

        self.ema_net = deepcopy(self.net)

    def forward(self, x):
        with torch.no_grad():
            x = self.ema_net(x)

        return x

    def update(self):
        state_dict = self.net.state_dict()

        if self.tau == 1:
            self.ema_net.load_state_dict(state_dict)
        elif self.tau > 0:
            update_sd = {k: self.tau * state_dict[k] + (1 - self.tau) * v
                         for k, v in self.ema_net.state_dict().items()}
            self.ema_net.load_state_dict(update_sd)


if __name__ == "__main__":
    pass
