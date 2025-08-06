import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray.tune.registry import register_env


class F1PitEnv(gym.Env):
    """
    A minimal env where:
      - obs = [current_lap_time, tire_wear, since_last_pit]
      - action = {0: stay out, 1: pit this lap}
      - reward = negative lap time (we want to minimize lap time + pit cost)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config):
        super().__init__()
        self.max_laps = config.get("max_laps", 70)
        self.current_lap = 0

        # Observation: [lap_time, tire_wear, since_last_pit]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32
        )
        # Action: stay out (0) or pit (1)
        self.action_space = spaces.Discrete(2)

        # state placeholders
        self.tire_wear = 0.0
        self.since_pit = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_lap = 0
        self.tire_wear = 0.0
        self.since_pit = 0
        obs = np.array(
            [self._lap_time(), self.tire_wear, self.since_pit], dtype=np.float32
        )
        info = {}
        return obs, info

    def step(self, action):
        done = False
        reward = 0.0

        if action == 1:
            reward -= 20.0
            self.tire_wear = 0.0
            self.since_pit = 0
        else:
            self.tire_wear += 0.01
            self.since_pit += 1

        lap_time = self._lap_time()
        reward -= lap_time

        self.current_lap += 1
        if self.current_lap >= self.max_laps:
            done = True

        obs = np.array([lap_time, self.tire_wear, self.since_pit], dtype=np.float32)
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    def _lap_time(self):
        return 90 + self.tire_wear * 50


def env_creator(config):
    return F1PitEnv(config)


register_env("f1-pit-env", env_creator)
