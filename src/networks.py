import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU(), skip_weight=0.5):
        super(ResBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.skip_weight = skip_weight

    def forward(self, x):
        return self.activation(self.linear(x)) + self.skip_weight * x


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100, activation=nn.ReLU()):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.resblock1 = ResBlock(hidden_dim, hidden_dim, activation)
        self.resblock2 = ResBlock(hidden_dim, hidden_dim, activation)
        self.resblock3 = ResBlock(hidden_dim, hidden_dim, activation)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.input_layer(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        return self.output_layer(x)


# Networks for the GAN-like mean-field game control formulation.
# NOmega approximates the value function (phi network)
class NOmega(nn.Module):
    def __init__(self):
        super(NOmega, self).__init__()
        # Input: 3 (state) + 1 (time) = 4; Output: scalar
        self.net = ResNet(input_dim=4, output_dim=1, activation=nn.Tanh())

    def forward(self, x, t):
        input_data = torch.cat([x, t], dim=-1)
        return self.net(input_data)


# NTheta approximates the generator.
class NTheta(nn.Module):
    def __init__(self):
        super(NTheta, self).__init__()
        # Input: 3 (latent) + 1 (time) = 4; Output: 3 (state)
        self.net = ResNet(input_dim=4, output_dim=3, activation=nn.Tanh())

    def forward(self, z, t):
        input_data = torch.cat([z, t], dim=-1)
        return self.net(input_data)

def phi_omega(x, t, N_omega, terminal_cost_fn):
    """
    Constructs the value function with boundary condition:
    φ_ω(x, t) = (1 - t) * N_omega(x, t) + t * g(x)
    """
    return (1 - t) * N_omega(x, t) + t * terminal_cost_fn(x)


def G_theta(z, t, N_theta:NTheta):
    """
    Constructs the generator with boundary condition:
    G_θ(z, t) = (1 - t) * z + t * N_theta(z, t)
    """
    return (1 - t) * z + t * N_theta(z, t)