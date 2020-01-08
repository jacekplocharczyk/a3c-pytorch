import torch
import gym
import pickle as pkl

from a3c_pytorch.common.basic_model import Actor, Critic
from a3c_pytorch.common.basic_memory import Memory
from a3c_pytorch.common.basic_logs import StatsLogger


ENV_NAME = "Pendulum-v0"
ROLLOUTS = 5000
GAMMA = 0.99
A_LR = 1e-3
C_LR = 1e-4
STATS_FREQ = 50
BATCH_SIZE = 512
REWARD_DONE = 190.0
NORMALIZE_ADV = True
LOGFILE = 'a2c_logs.pkl'
TD = False


def main():
    env = gym.make(ENV_NAME)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # model = ActorCritic(ob_dim, ac_dim, discrete)
    actor = Actor(ob_dim, ac_dim, discrete)
    critic = Critic(ob_dim)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=A_LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=C_LR)
    logger = StatsLogger()
    stats = []

    for i in range(ROLLOUTS):
        memory = perform_rollout(env, actor, critic, gamma=GAMMA)

        update_ac(actor, critic, actor_optimizer, critic_optimizer, memory)

        running_reward = logger.get_running_reward(memory)

        if not i % STATS_FREQ:
            logger.rollout_stats(memory, i)
            stats.append(logger.get_rollout_stats(memory, i))

        if running_reward > REWARD_DONE:
            logger.task_done(i)
            break

        with open(LOGFILE, 'wb') as f:
            pkl.dump(stats, f)


def perform_rollout(
    env: gym.Env,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    gamma: float) -> Memory:
    memory = Memory(gamma, BATCH_SIZE)
    obs = env.reset()
    done = False
    while not done:
        obs = torch.unsqueeze(torch.FloatTensor(obs), dim=0)
        action, action_logprobs = actor.act(obs)
        if TD:
            raise NotImplementedError
        else:
            state_value = critic(obs)

        obs, rew, done, _ = env.step(action)

        memory.update_actions(action)
        memory.update_action_logprobs(action_logprobs)
        memory.update_state_values(state_value)
        memory.update_rewards(torch.tensor(rew))
        memory.update_is_terminals(torch.tensor(done, dtype=torch.uint8))
    return memory


def update_ac(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    memory: Memory,
) -> torch.nn.Module:

    memory.calculate_returns()

    for batch in memory:

        # Step 1: Update critic

        # Step 2: Estimate advantage

        # Step 3: Update actor
        action_logprobs = torch.squeeze(torch.stack(batch.action_logprobs))
        state_values = torch.squeeze(torch.stack(batch.state_values))
        returns = torch.squeeze(batch.returns)

        advantage = returns - state_values
        if NORMALIZE_ADV:
            advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-8)

        critic_loss = 0.5 * advantage.pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        advantage_no_grad = advantage.detach()
        actor_loss = (-action_logprobs * advantage_no_grad).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()


if __name__ == "__main__":
    main()
