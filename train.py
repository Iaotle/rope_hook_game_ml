import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time

# Assume RopeEnv is defined as in the previous conversion,
# and it provides an observation vector that includes:
# [anchor_x, anchor_y, ball_x, ball_y, enemy1_rel_x, enemy1_rel_y,
#  enemy2_rel_x, enemy2_rel_y, enemy3_rel_x, enemy3_rel_y,
#  enemy4_rel_x, enemy4_rel_y, enemy5_rel_x, enemy5_rel_y,
#  ball_velocity, ball_angle, relative_angles, ... , score]
# For demonstration, we assume observation_dim = 20 (change as needed).
# OBSERVATION_DIM = 20   # Change this if you add more features.
# ACTION_DIM = 2         # Continuous action: (dx, dy)

# Set device to GPU if available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super(ActorCritic, self).__init__()
        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Actor head: outputs mean of action distribution.
        self.actor_mean = nn.Linear(hidden_size, act_dim)
        # Log std is maintained as a parameter (independent of state).
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        # Critic head: outputs state value.
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        action_mean = self.actor_mean(shared_out)
        action_std = torch.exp(self.log_std)
        value = self.critic(shared_out)
        return action_mean, action_std, value

    def get_action(self, x):
        action_mean, action_std, value = self.forward(x)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, x, action):
        action_mean, action_std, value = self.forward(x)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_param=0.2,
                 ppo_epochs=10, batch_size=64):
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.policy = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Buffer to store transitions
        self.memory = []

    def store_transition(self, transition):
        # transition: (obs, action, log_prob, reward, done, value)
        self.memory.append(transition)

    def compute_returns_and_advantages(self, last_value, dones):
        rewards = [t[3] for t in self.memory]
        values = [t[5] for t in self.memory]
        returns = []
        advs = []
        R = last_value
        for r, done, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        values = torch.tensor(values, dtype=torch.float32, device=device)
        advantages = returns - values
        return returns, advantages

    def update(self):
        # Unpack transitions from memory.
        # Stack numpy arrays for obs and actions before converting to tensor
        obs_np = np.stack([t[0] for t in self.memory])
        actions_np = np.stack([t[1] for t in self.memory])
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
        actions = torch.tensor(actions_np, dtype=torch.float32, device=device)
        old_log_probs = torch.tensor([t[2] for t in self.memory], dtype=torch.float32, device=device)
        rewards = [t[3] for t in self.memory]
        dones = [t[4] for t in self.memory]
        values = [t[5] for t in self.memory]

        # Get last value for advantage calculation (assume not done)
        with torch.no_grad():
            _, _, last_value = self.policy(obs[-1].unsqueeze(0))
            last_value = last_value.item()

        returns, advantages = self.compute_returns_and_advantages(last_value, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(self.memory)
        indices = np.arange(dataset_size)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                log_probs, entropy, values_new = self.policy.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (mb_returns - values_new.squeeze()).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory = []  # Clear memory after update

# ----------------------------
# Training Loop using RopeEnv
# ----------------------------

from RopeDefenseEnv import RopeEnv
def train_ppo(num_updates=1000, update_interval=2048):
    env = RopeEnv(render_mode=False)
    # get dimensions from env
    OBSERVATION_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    
    agent = PPOAgent(obs_dim=OBSERVATION_DIM, act_dim=ACTION_DIM)
    obs = env.reset()
    episode_rewards = []
    total_steps = 0
    episode_reward = 0

    start_time = time.time()
    for update in range(num_updates):
        for _ in range(update_interval):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = agent.policy.get_action(obs_tensor)
            action = action.squeeze(0).cpu().numpy()
            new_obs, reward, done, _ = env.step(action)
            agent.store_transition((obs, action, log_prob.item(), reward, done, value.item()))
            episode_reward += reward
            total_steps += 1
            obs = new_obs

            if done:
                obs = env.reset()
                episode_rewards.append(episode_reward)
                episode_reward = 0

        agent.update()
        # if update % 10 == 0 and episode_rewards:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Update {update}, Steps {total_steps}, Avg Episode Reward: {avg_reward:.2f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    env.close()

if __name__ == '__main__':
    train_ppo()