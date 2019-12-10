""""
CartPole-v0 version.
"""

from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.softmax = nn.Softmax(dim=1)
        self.act_fc = nn.Linear(64, 2)
        self.cri_fc = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def actor(self, x):
        x = self.softmax(self.act_fc(x))
        return x

    def critic(self, x):
        return self.cri_fc(x)

    def act(self, obs):
        action_prob, state_value = self.two_heads_return(obs)
        dist = Categorical(action_prob)
        action = dist.sample()
        action_logprobs = dist.log_prob(torch.squeeze(action))

        return action, action_logprobs, state_value

    def two_heads_return(self, obs) -> Tuple[torch.tensor, torch.tensor]:
        forward = self.forward(obs)
        action_prob = self.actor(forward)
        state_value = self.critic(forward)
        return action_prob, state_value
