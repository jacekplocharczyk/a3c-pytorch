import datetime
from pathlib import Path

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from a3c_pytorch.common.basic_memory import Memory

TENSORBOARD_DIR = Path("tensorboard")


class TensorboardWriter(SummaryWriter):
    def __init__(
        self,
        device: torch.device,
        lr: float,
        comment: str = "",
        histograms: bool = True,
    ):
        log_dir = self.get_log_dir(device, lr, comment)
        self.use_histograms = True
        super().__init__(log_dir=log_dir)

    @staticmethod
    def get_log_dir(device: torch.device, lr: float, comment: str) -> Path:
        """
        Get tensorboard log dir based on params
        Arguments:
            device {torch.device} -- GPU or CPU device
            lr {float} -- learning rate
            comment {str} -- user comment
        Returns:
            Path -- log dir
        """

        dev_str = str(device).split(":")[0]
        tensorboard_suffix = f"_dev{dev_str}_lr{lr}" + comment
        current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = TENSORBOARD_DIR / (current_time + "_" + tensorboard_suffix)
        return log_dir

    def rollout_stats(self, memory: Memory, i: int):
        if self.use_histograms:
            self._add_memory_histograms(memory, i)
        self._add_reward_plot(memory, i)

    def _add_memory_histograms(self, memory: Memory, i: int):
        actions = torch.squeeze(torch.stack(memory.actions))
        action_logprobs = torch.squeeze(torch.stack(memory.action_logprobs))
        state_values = torch.squeeze(torch.stack(memory.state_values))
        returns = torch.squeeze(memory.returns)

        self.add_histogram("Actions", actions, i)
        self.add_histogram("Actions logprobs", action_logprobs, i)
        self.add_histogram("State values", state_values, i)
        self.add_histogram("Returns", returns, i)

    def _add_reward_plot(self, memory: Memory, i: int):
        full_reward = memory.get_summary_reward()
        self.add_scalar("Episode reward", full_reward, i)
