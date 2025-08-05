import gym
import numpy as np
from gym import spaces
from ray.tune.registry import register_env


class F1PitEnv(gym.Env):
    """
    A minimal env where:
      - obs = [current_lap_time, tire_wear, last_pit_lap]
      - action = {0: stay out, 1: pit this lap}
      - reward = negative lap time (we want to minimize lap time + pit cost)
    """

    def __init__(self, config):
        super().__init__()
        # config can hold track metadata, tire degradation curves, etc.
        self.max_laps = config.get("max_laps", 70)
        self.current_lap = 0

        # Observation: [lap_time, tire_wear, since_last_pit]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32
        )
        # Action: stay (0) or pit (1)
        self.action_space = spaces.Discrete(2)

        # state placeholders
        self.tire_wear = 0.0
        self.since_pit = 0

    def reset(self):
        self.current_lap = 0
        self.tire_wear = 0.0
        self.since_pit = 0
        # initial lap time baseline
        obs = np.array(
            [self._lap_time(), self.tire_wear, self.since_pit], dtype=np.float32
        )
        return obs

    def step(self, action):
        done = False
        reward = 0.0

        # if we pit
        if action == 1:
            # reset wear, penalty for pit delta
            reward -= 20.0
            self.tire_wear = 0.0
            self.since_pit = 0
        else:
            # accumulate wear
            self.tire_wear += 0.01
            self.since_pit += 1

        # simulate next lap
        lap_time = self._lap_time()
        reward -= lap_time  # lower lap time → higher reward

        self.current_lap += 1
        if self.current_lap >= self.max_laps:
            done = True

        obs = np.array([lap_time, self.tire_wear, self.since_pit], dtype=np.float32)
        info = {}
        return obs, reward, done, info

    def _lap_time(self):
        # base 90s plus wear penalty
        return 90 + self.tire_wear * 50


def env_creator(config):
    return F1PitEnv(config)


register_env("f1-pit-env", env_creator)
