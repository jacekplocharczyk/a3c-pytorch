import torch

from a3c_pytorch.common.basic_memory import Memory, MemoryBatch


class MultiprocessMemory(Memory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = []
        self.rewards = []
        self.is_terminals = []

    def update_states(self, value: torch.tensor):
        self.states.append(value)

    def update_is_terminals(self, value: torch.tensor):
        self.is_terminals.append(value)

    def update_rewards(self, value: torch.tensor):
        self.rewards.append(value)

    def __add__(self, other: Memory):
        if self.gamma != other.gamma or self.batch_size != other.batch_size:
            raise ValueError("Memories have different gammas or batch sizes.")
        new_memory = MultiprocessMemory(self.gamma, self.batch_size)

        new_memory.actions = self.actions + other.actions
        new_memory.states = self.states + other.states
        new_memory.rewards = self.rewards + other.rewards
        new_memory.is_terminals = self.is_terminals + other.is_terminals

        return new_memory

    def __next__(self):  # Python 2: def next(self)
        start = self.batch_size * self.batch
        end = start + self.batch_size

        if start >= self.returns.shape[0]:
            raise StopIteration
        self.batch += 1
        return MultiprocessMemoryBatch(self, start, end)


class MultiprocessMemoryBatch(MemoryBatch):
    def __init__(self, memory: Memory, start: int, end: int):
        super().__init__(memory, start, end)
        self.states = memory.states[start:end]
        self.actions = memory.actions[start:end]
