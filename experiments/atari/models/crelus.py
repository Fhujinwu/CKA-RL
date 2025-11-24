import torch
import torch.nn as nn
import torch.nn.functional as F


def crelu(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """CReLU activation function.

    CReLU is the concatenation of ReLU applied to x and -x: CReLU(x) = [ReLU (x), ReLU (-x)].
    Applying CReLU to a tensor of shape (N, D) results in a tensor of shape (N, 2D).

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    dim : int
        Dimension along which the features will be concatenated. Default is 1.

    Returns
    -------
    torch.Tensor
        Tensor with CReLU applied to input.
    """

    return torch.cat((F.relu(input), F.relu(-input)), dim=dim)


class CReLU(nn.Module):
    """CReLU activation function.

    CReLU is the concatenation of ReLU applied to x and -x: CReLU(x) = [ReLU (x), ReLU (-x)].
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Apply the CReLU function to the input tensor.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.
        dim : int
            Dimension along which the features will be concatenated. Default is 1.

        Returns
        -------
        torch.Tensor
            Tensor with CReLU applied to input.
        """
        return crelu(input, dim)


if __name__ == "__main__":
    c = CReLU()

    # Test that it works for common liner use case
    t1 = torch.randn(32, 64)
    linear = nn.Linear(64, 64)

    assert c(t1).shape == (32, 128), "Invalid shape for CReLU linear output."


import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from .cnn_encoder import CnnEncoder


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CReLUsAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = CnnEncoder(hidden_dim=512, layer_init=layer_init)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, 256)),
            CReLU(),
            layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01),
        )
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.actor, f"{dirname}/actor.pt")
        torch.save(self.network, f"{dirname}/encoder.pt")
        torch.save(self.critic, f"{dirname}/critic.pt")

    def load(dirname, envs, load_critic=True, reset_actor=False, map_location=None):
        model = CReLUsAgent(envs)
        model.network = torch.load(f"{dirname}/encoder.pt", map_location=map_location)
        if not reset_actor:
            model.actor = torch.load(f"{dirname}/actor.pt", map_location=map_location)
        if load_critic:
            model.critic = torch.load(f"{dirname}/critic.pt", map_location=map_location)
        return model
