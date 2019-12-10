from a3c_pytorch.common.basic_memory import Memory


class StatsLogger:
    def __init__(self, alpha: float = 0.9):
        self.running_reward = None
        self.alpha = alpha

    def get_running_reward(self, memory: Memory) -> float:
        new_mean_reward = float(memory.get_summary_reward())
        if self.running_reward is None:
            self.running_reward = new_mean_reward
        else:
            self.running_reward *= self.alpha
            self.running_reward += (1 - self.alpha) * new_mean_reward
        return self.running_reward

    def rollout_stats(self, memory: Memory, rollout: int) -> None:
        print(f"Rollout {rollout:5}\tRunning reward: {self.running_reward}")

    def task_done(self, i: int) -> None:
        if str(i)[-1] == "1":
            rollout = str(i) + "st"
        elif str(i)[-1] == "2":
            rollout = str(i) + "nd"
        elif str(i)[-1] == "3":
            rollout = str(i) + "rd"
        else:
            rollout = str(i) + "th"

        print(
            f"Task finished at {rollout} rollout. "
            f"Running reward is {self.running_reward}"
        )
