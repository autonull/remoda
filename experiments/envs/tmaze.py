import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TMazeEnv(gym.Env):
    """
    One-Shot T-Maze Environment.

    A sequence of N rooms.
    Room 0: Signpost (0=Left, 1=Right)
    Rooms 1 to N-2: Hallway
    Room N-1: The T-Junction. Action must match the signpost.

    Observation space:
    [is_signpost, signpost_value, is_hallway, is_junction]
    """
    def __init__(self, n_rooms=50):
        super().__init__()
        self.n_rooms = n_rooms
        self.current_room = 0
        self.signpost = 0

        # Actions: 0=Left, 1=Right
        self.action_space = spaces.Discrete(2)

        # Obs: [is_signpost, signpost_val, is_hallway, is_junction]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_room = 0
        self.signpost = self.np_random.integers(0, 2)
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(4, dtype=np.float32)
        if self.current_room == 0:
            obs[0] = 1.0
            obs[1] = float(self.signpost)
        elif self.current_room == self.n_rooms - 1:
            obs[3] = 1.0
        else:
            obs[2] = 1.0
        return obs

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self.current_room == self.n_rooms - 1:
            terminated = True
            if action == self.signpost:
                reward = 10.0
            else:
                reward = -10.0
        else:
            self.current_room += 1
            # Small penalty for time
            reward = -0.1

        return self._get_obs(), reward, terminated, truncated, info
