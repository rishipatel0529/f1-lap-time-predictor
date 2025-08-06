import gymnasium as gym
import numpy as np
from gymnasium import spaces


class F1PitEnv(gym.Env):
    """
    - obs = [lap_time, tire_wear, since_last_pit]
    - action = 0 (stay out) or 1 (pit this lap)
    - reward = -(lap_time) - (pit?penalty) + (finish_bonus)
    """

    def __init__(self, config):
        super().__init__()
        self.max_laps = config.get("max_laps", 70)
        self.pit_penalty = config.get("pit_penalty", 10.0)  # reduced penalty
        self.finish_bonus = config.get("finish_bonus", 100.0)  # bonus for finishing
        self.current_lap = 0

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self.tire_wear = 0.0
        self.since_pit = 0

    def reset(self):
        self.current_lap = 0
        self.tire_wear = 0.0
        self.since_pit = 0
        obs = np.array(
            [self._lap_time(), self.tire_wear, self.since_pit], dtype=np.float32
        )
        return obs, {}

    def step(self, action):
        reward = 0.0
        # pit or not
        if action == 1:
            self.tire_wear = 0.0
            self.since_pit = 0
            reward -= self.pit_penalty
        else:
            self.tire_wear += 0.01
            self.since_pit += 1

        lap_time = self._lap_time()
        reward -= lap_time

        self.current_lap += 1
        done = self.current_lap >= self.max_laps
        if done:
            reward += self.finish_bonus

        obs = np.array([lap_time, self.tire_wear, self.since_pit], dtype=np.float32)
        return obs, reward, done, False, {}

    def _lap_time(self):
        return 90.0 + self.tire_wear * 50.0


def env_creator(config):
    return F1PitEnv(config)
