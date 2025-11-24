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
    
import torch
import torch.nn as nn
import os

def Cshared(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        CReLU(),
        nn.Linear(256, 128),
        CReLU(),
    )

class CReLUsAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        self.fc = Cshared(input_dim=obs_dim)

        # will be created when calling `reset_heads`
        self.fc_mean = None
        self.fc_logstd = None
        self.reset_heads()

    def reset_heads(self):
        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

    def forward(self, x):
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/model.pt")

    def load(dirname, map_location=None, reset_heads=False):
        model = torch.load(f"{dirname}/model.pt", map_location=map_location)
        if reset_heads:
            model.reset_heads()
        return model
