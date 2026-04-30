import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DelayedMatchToSampleEnv(gym.Env):
    """
    Delayed Match-to-Sample Environment.

    A sample is shown, followed by a delay of D steps.
    Then, the agent must choose the action corresponding to the sample.

    Observation space:
    [is_sample, sample_class, is_delay, is_decision]
    """
    def __init__(self, delay_length=50, num_classes=3):
        super().__init__()
        self.delay_length = delay_length
        self.num_classes = num_classes
        self.current_step = 0
        self.sample = 0

        # Actions: Choose the class (0 to num_classes-1)
        self.action_space = spaces.Discrete(num_classes)

        # Obs: [is_sample, one_hot_sample..., is_delay, is_decision]
        self.obs_dim = 1 + num_classes + 1 + 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.sample = self.np_random.integers(0, self.num_classes)
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        if self.current_step == 0:
            obs[0] = 1.0
            obs[1 + self.sample] = 1.0
        elif self.current_step == self.delay_length + 1:
            obs[-1] = 1.0  # is_decision
        else:
            obs[-2] = 1.0  # is_delay
        return obs

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self.current_step == self.delay_length + 1:
            terminated = True
            if action == self.sample:
                reward = 1.0
            else:
                reward = -1.0
        else:
            self.current_step += 1

        return self._get_obs(), reward, terminated, truncated, info
