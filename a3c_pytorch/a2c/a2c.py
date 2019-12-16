import torch
import gym

from a3c_pytorch.common.basic_model import ActorCritic
from a3c_pytorch.common.basic_memory import Memory
from a3c_pytorch.common.basic_logs import StatsLogger


ENV_NAME = "Pendulum-v0"
ROLLOUTS = 5000
GAMMA = 0.99
LR = 3e-3
STATS_FREQ = 100
BATCH_SIZE = 128
REWARD_DONE = 190.0


def main():
    env = gym.make(ENV_NAME)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    model = ActorCritic(ob_dim, ac_dim, discrete)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    logger = StatsLogger()

    for i in range(ROLLOUTS):
        memory = perform_rollout(env, model, gamma=GAMMA)

        model = update_model(model, optimizer, memory)

        running_reward = logger.get_running_reward(memory)

        if not i % STATS_FREQ:
            logger.rollout_stats(memory, i)

        if running_reward > REWARD_DONE:
            logger.task_done(i)
            break


def perform_rollout(env: gym.Env, model: torch.nn.Module, gamma: float) -> Memory:
    memory = Memory(gamma, BATCH_SIZE)
    obs = env.reset()
    done = False

    while not done:
        obs = torch.unsqueeze(torch.FloatTensor(obs), dim=0)
        action, action_logprobs, state_value = model.act(obs)

        obs, rew, done, _ = env.step(action)

        memory.update_actions(action)
        memory.update_action_logprobs(action_logprobs)
        memory.update_state_values(state_value)
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

        action_logprobs = torch.squeeze(torch.stack(batch.action_logprobs))
        state_values = torch.squeeze(torch.stack(batch.state_values))
        returns = torch.squeeze(batch.returns)

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
