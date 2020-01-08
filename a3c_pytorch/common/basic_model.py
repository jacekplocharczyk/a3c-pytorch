""""
CartPole-v0 version.
"""

from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical, Normal, Independent


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, ac_dim: int, discrete: bool = True):
        super().__init__()

        self.ac_dim = ac_dim
        self.discrete = discrete
        self.fc1 = nn.Linear(obs_dim, 16)
        self.fc2 = nn.Linear(obs_dim, 16)
        self.softmax = nn.Softmax(dim=1)
        self.act_fc = nn.Linear(16, ac_dim)
        self.cri_fc = nn.Linear(16, 1)
        if not self.discrete:
            self.log_scale = nn.Parameter(0.5 * torch.ones(self.ac_dim), requires_grad=True)

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     return x

    def actor(self, x):
        x = F.relu(self.fc1(x))
        if self.discrete:
            x = self.softmax(self.act_fc(x))
        else:
            x = 2 * torch.tanh(self.act_fc(x))
        return x

    def critic(self, x):
        x = F.relu(self.fc1(x))
        return self.cri_fc(x)

    def act(self, obs):
        action_prob, state_value = self.two_heads_return(obs)
        if self.discrete:
            dist = Categorical(action_prob)
        else:
            normal = Normal(action_prob, torch.exp(self.log_scale))
            dist = Independent(normal, 1)
        action = dist.sample()
        action_logprobs = dist.log_prob(torch.squeeze(action))

        return action, action_logprobs, state_value  # TODO: squeeze state_value

    def two_heads_return(self, obs) -> Tuple[torch.tensor, torch.tensor]:
        # forward = self.forward(obs)
        action_prob = self.actor(obs)
        state_value = self.critic(obs)
        return action_prob, state_value

class Actor(nn.Module):
    def __init__(self, obs_dim: int, ac_dim: int, discrete: bool = True):
        super().__init__()

        self.ac_dim = ac_dim
        self.obs_dim = obs_dim
        self.discrete = discrete
        self.fc1 = nn.Linear(obs_dim, 16)
        self.fc2 = nn.Linear(16, ac_dim)
        if not self.discrete:
            self.log_scale = nn.Parameter(0.5 * torch.ones(self.ac_dim), requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.discrete:
            x = self.softmax(self.fc2(x))
        else:
            x = 2 * torch.tanh(self.fc2(x))
        return x

    def act(self, obs):
        action_prob = self.forward(obs)
        if self.discrete:
            dist = Categorical(action_prob)
        else:
            normal = Normal(action_prob, torch.exp(self.log_scale))
            dist = Independent(normal, 1)
        action = dist.sample()
        action_logprobs = dist.log_prob(torch.squeeze(action))

        return action, action_logprobs

class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x