import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from train import train_actor_critic
from envs import TMazeEnv, DelayedMatchToSampleEnv
from pgra.models import BaselineActorCritic, PGRAActorCritic

def smooth(scalars, weight=0.9):
    """
    EMA smoothing.
    """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def run_experiment(env_name, num_episodes=500, seeds=[42, 100, 1234]):
    if env_name == "tmaze":
        env_cls = TMazeEnv
        env_kwargs = {'n_rooms': 15}
        obs_dim = 4
        act_dim = 2
    elif env_name == "match_to_sample":
        env_cls = DelayedMatchToSampleEnv
        env_kwargs = {'delay_length': 15, 'num_classes': 3}
        obs_dim = 1 + 3 + 1 + 1 # 6
        act_dim = 3
    else:
        raise ValueError("Unknown env")

    all_baseline_rewards = []
    all_pgra_rewards = []

    for seed in seeds:
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"--- Running {env_name} with Baseline (Seed: {seed}) ---")
        baseline_rewards = train_actor_critic(
            env_cls=env_cls,
            env_kwargs=env_kwargs,
            model_cls=BaselineActorCritic,
            model_kwargs={'obs_dim': obs_dim, 'act_dim': act_dim, 'hidden_dim': 64},
            is_pgra=False,
            num_episodes=num_episodes
        )
        all_baseline_rewards.append(baseline_rewards)

        print(f"--- Running {env_name} with PGRA (Seed: {seed}) ---")
        pgra_rewards = train_actor_critic(
            env_cls=env_cls,
            env_kwargs=env_kwargs,
            model_cls=PGRAActorCritic,
            model_kwargs={'obs_dim': obs_dim, 'act_dim': act_dim, 'hidden_dim': 64},
            is_pgra=True,
            num_episodes=num_episodes
        )
        all_pgra_rewards.append(pgra_rewards)

    return np.array(all_baseline_rewards), np.array(all_pgra_rewards)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["tmaze", "match_to_sample", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()

    envs_to_run = ["tmaze", "match_to_sample"] if args.env == "both" else [args.env]

    os.makedirs("plots", exist_ok=True)

    for env_name in envs_to_run:
        b_rewards, p_rewards = run_experiment(env_name, num_episodes=args.episodes)

        b_mean = np.mean(b_rewards, axis=0)
        b_std = np.std(b_rewards, axis=0)
        p_mean = np.mean(p_rewards, axis=0)
        p_std = np.std(p_rewards, axis=0)

        plt.figure(figsize=(10, 6))

        # Baseline
        b_mean_smoothed = smooth(b_mean.tolist())
        plt.plot(b_mean_smoothed, color='blue', label='Baseline (LSTM)')
        plt.fill_between(range(len(b_mean_smoothed)), np.array(b_mean_smoothed) - b_std, np.array(b_mean_smoothed) + b_std, color='blue', alpha=0.2)

        # PGRA
        p_mean_smoothed = smooth(p_mean.tolist())
        plt.plot(p_mean_smoothed, color='orange', label='PGRA (v2.1)')
        plt.fill_between(range(len(p_mean_smoothed)), np.array(p_mean_smoothed) - p_std, np.array(p_mean_smoothed) + p_std, color='orange', alpha=0.2)

        plt.title(f"Learning Curve: {env_name.replace('_', ' ').title()}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = f"plots/{env_name}_learning_curve.png"
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
