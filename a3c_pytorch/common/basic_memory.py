from typing import Union

import torch


class Memory:
    def __init__(self, gamma: float, batch_size: int = 64):
        self.actions = []
        self.action_logprobs = []
        self.state_values = []
        self.returns = None

        self.rewards = None
        self.is_terminals = None
        self.batch = 0
        self.gamma = gamma
        self.batch_size = batch_size

    @staticmethod
    def _append(
        memory_variable: Union[None, torch.tensor], value: torch.tensor
    ) -> torch.tensor:
        value = torch.unsqueeze(value, dim=0)[None, :]
        if memory_variable is None:
            return value
        else:
            return torch.cat([memory_variable, value])  # WARN: cat loses grad_fn!

    def update_actions(self, value: torch.tensor):
        self.actions.append(value)

    def update_action_logprobs(self, value: torch.tensor):
        self.action_logprobs.append(value)

    def update_state_values(self, value: torch.tensor):
        self.state_values.append(value)

    def update_rewards(self, value: torch.tensor):
        self.rewards = self._append(self.rewards, value)

    def update_is_terminals(self, value: torch.tensor):
        self.is_terminals = self._append(self.is_terminals, value)

    def calculate_returns(self):
        returns = []
        discounted_return = 0

        for reward, is_terminal in zip(
            reversed(self.rewards), reversed(self.is_terminals)
        ):
            if is_terminal:
                discounted_return = 0
            discounted_return = reward + (self.gamma * discounted_return)
            returns.insert(0, discounted_return)

        self.returns = torch.tensor(returns)

    def __iter__(self):
        self.batch = 0
        return self

    def __next__(self):  # Python 2: def next(self)
        start = self.batch_size * self.batch
        end = start + self.batch_size

        if start >= self.returns.shape[0]:
            raise StopIteration
        self.batch += 1
        return MemoryBatch(self, start, end)

    def get_summary_reward(self):
        return float(self.rewards.sum()) / float(sum(self.is_terminals))


class MemoryBatch:
    def __init__(self, memory: Memory, start: int, end: int):
        self.action_logprobs = memory.action_logprobs[start:end]
        self.state_values = memory.state_values[start:end]
        self.returns = memory.returns[start:end]
