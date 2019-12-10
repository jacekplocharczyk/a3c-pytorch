import torch
from torch.distributions import Categorical

from a3c_pytorch.common.basic_model import ActorCritic


class MultiprocessActorCritic(ActorCritic):
    def act(self, obs):
        if self.training:
            self.eval()
        with torch.no_grad():
            forward = self.forward(obs)
            action_prob = self.actor(forward)
            dist = Categorical(action_prob)
            action = dist.sample()
        return action

    def evaluate(self, obs, action):
        if not self.training:
            self.train()

        action_prob, state_value = self.two_heads_return(obs)
        action_prob = torch.squeeze(action_prob)
        dist = Categorical(action_prob)

        action_logprobs = dist.log_prob(torch.squeeze(action))

        return action_logprobs, torch.squeeze(state_value)
