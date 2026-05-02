import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

def run_episode(env, model, is_pgra, device="cpu", render=False):
    obs, _ = env.reset()
    done = False

    # Init recurrent state
    # batch_size=1
    state = model.init_state(1, device)

    log_probs = []
    values = []
    rewards = []
    entropies = []

    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # We need to construct an external signal for PGRA.
        # In a real setup, we'd use TD error or reward. Since reward comes AFTER action,
        # we can use the previous reward or a simple surprise metric.
        # For simplicity in this demo, we'll feed a 0 external signal during the episode,
        # and let the differentiable saliency mechanism learn when to trigger.
        ext_signal = torch.zeros((1, 1), device=device)

        if is_pgra:
            logits, value, state = model(x, state, external_signal=ext_signal)
        else:
            logits, value, state = model(x, state)

        dist = Categorical(logits=logits)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        entropies.append(entropy)

    return log_probs, values, rewards, entropies

def train_actor_critic(env_cls, env_kwargs, model_cls, model_kwargs, is_pgra, num_episodes=500, lr=1e-3, gamma=0.99, entropy_coef=0.01):
    env = env_cls(**env_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_cls(**model_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    episode_rewards = []

    for ep in range(num_episodes):
        log_probs, values, rewards, entropies = run_episode(env, model, is_pgra, device=device)

        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=device)

        # Standardize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        value_loss = []

        for log_prob, value, R, entropy in zip(log_probs, values, returns, entropies):
            advantage = R - value.item()

            policy_loss.append(-log_prob * advantage - entropy_coef * entropy)
            value_loss.append(nn.functional.mse_loss(value, torch.tensor([[R]], device=device)))

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()

        # We need retain_graph=True because PGRA traces (E_t, V_sal_t) might hold references to graph,
        # but in detached forward it should be fine. Let's just catch if we need it.
        try:
            loss.backward()
        except RuntimeError:
            loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{num_episodes} | Reward: {ep_reward:.2f}")

    return episode_rewards

if __name__ == "__main__":
    from envs import TMazeEnv
    from pgra.models import BaselineActorCritic, PGRAActorCritic

    print("Testing train script with BaselineActorCritic...")
    train_actor_critic(
        env_cls=TMazeEnv,
        env_kwargs={'n_rooms': 10},
        model_cls=BaselineActorCritic,
        model_kwargs={'obs_dim': 4, 'act_dim': 2, 'hidden_dim': 32},
        is_pgra=False,
        num_episodes=10
    )
    print("Done testing.")
