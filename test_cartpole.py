import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")
obs, _ = env.reset()
done = False
total_reward = 0
while not done:
    pole_angle = obs[2]
    action = 1 if pole_angle > 0 else 0
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Total reward with heuristic: {total_reward}")
