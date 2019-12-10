import copy

import gym
import torch

from a3c_pytorch.common.basic_logs import StatsLogger
from a3c_pytorch.common.multiprocess_memory import MultiprocessMemory as Memory
from a3c_pytorch.common.multiprocess_model import MultiprocessActorCritic as ActorCritic

torch.multiprocessing.set_start_method(
    "spawn", True
)  # required for debugging in vscode

ENV_NAME = "CartPole-v0"
OBS_DIM = 4
PER_CORE_ROLLOUTS = 5000
CORES = 6
GAMMA = 0.99
LR = 3e-4
STATS_FREQ = 1
BATCH_SIZE = 128
REWARD_DONE = 190.0


def main():
    model = ActorCritic(OBS_DIM)
    logger = StatsLogger()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for i in range(PER_CORE_ROLLOUTS):
        arguments = [
            (
                ENV_NAME,
                copy.deepcopy(model),
                GAMMA,
                BATCH_SIZE,
                Memory(GAMMA, BATCH_SIZE),
            )
        ] * CORES

        memory = Memory(GAMMA, BATCH_SIZE)
        with torch.multiprocessing.Pool(CORES) as p:
            memories = p.map(perform_rollout, arguments)

        for m in memories:
            memory = memory + m
            del m

        model = update_model(model, optimizer, memory)

        running_reward = logger.get_running_reward(memory)

        if not i % STATS_FREQ:
            logger.rollout_stats(memory, i)

        if running_reward > REWARD_DONE:
            logger.task_done(i)
            break


def perform_rollout(args: tuple) -> Memory:
    env_name, model, gamma, batch_size, memory = args
    env = gym.make(env_name)

    # memory = Memory(gamma, batch_size)
    obs = env.reset()
    done = False

    while not done:

        obs = torch.unsqueeze(torch.FloatTensor(obs), dim=0)
        action = model.act(obs)
        memory.update_states(obs)

        obs, rew, done, _ = env.step(int(action))

        memory.update_actions(action)
        memory.update_rewards(torch.tensor(rew))
        memory.update_is_terminals(torch.tensor(done, dtype=torch.uint8))

    return memory


def update_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    memory: Memory,
    lr: float = LR,
) -> torch.nn.Module:

    memory.calculate_returns()

    for batch in memory:

        actions = torch.squeeze(torch.stack(batch.actions))
        states = torch.squeeze(torch.stack(batch.states))
        returns = torch.squeeze(batch.returns)

        action_logprobs, state_values = model.evaluate(states, actions)

        advantage = returns - state_values

        critic_loss = 0.5 * advantage.pow(2).mean()
        actor_loss = (-action_logprobs * advantage).mean()

        cumulated_loss = actor_loss + critic_loss

        optimizer.zero_grad()

        cumulated_loss.backward()
        optimizer.step()

    return model


if __name__ == "__main__":
    main()
