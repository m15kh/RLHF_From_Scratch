import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariA2C(nn.Module):
    """Actor-critic network: shared conv body, separate policy and value heads."""
    def __init__(self, input_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.policy = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: uint8 (B, 4, 84, 84) -> (policy logits (B, n_actions), values (B,))"""
        conv_out = self.conv(x.float() / 255.0)
        return self.policy(conv_out), self.value(conv_out).squeeze(-1)


class RewardModel(nn.Module):
    """
    Reward predictor r_hat(obs, action) from "Deep RL from Human Preferences"
    (Christiano et al. 2017, appendix A.1): 4 conv layers (7x7/3, 5x5/2, 3x3/1, 3x3/1),
    16 filters each, batch norm + dropout 0.5 + leaky ReLU, then FC 64 and a scalar output.
    The action is fed as a one-hot vector next to the conv features.
    """
    def __init__(self, input_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        self.n_actions = n_actions

        def block(c_in: int, c_out: int, kernel: int, stride: int) -> list[nn.Module]:
            return [
                nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=stride),
                nn.BatchNorm2d(c_out),
                nn.Dropout(p=0.5),
                nn.LeakyReLU(0.01),
            ]

        self.conv = nn.Sequential(
            *block(input_shape[0], 16, 7, 3),
            *block(16, 16, 5, 2),
            *block(16, 16, 3, 1),
            *block(16, 16, 3, 1),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(2, *input_shape)).size()[-1]
        self.out = nn.Sequential(
            nn.Linear(size + n_actions, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        """obs: uint8 (N, 4, 84, 84), acts: int64 (N,) -> predicted reward (N,)"""
        conv_out = self.conv(obs.float() / 255.0)
        acts_oh = F.one_hot(acts, self.n_actions).float()
        return self.out(torch.cat([conv_out, acts_oh], dim=1)).squeeze(-1)
