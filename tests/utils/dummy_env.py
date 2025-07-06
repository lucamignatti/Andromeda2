import numpy as np

class DummyEnv:
    """
    A dummy environment that generates random data for testing the training loop.
    """
    def __init__(self, observation_shape, action_dim, reward_dim):
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.reward_dim = reward_dim

    def reset(self):
        return np.random.randn(*self.observation_shape).astype(np.float32)

    def step(self, action):
        """
        Returns a random (next_observation, reward, done, info) tuple.
        """
        next_obs = np.random.randn(*self.observation_shape).astype(np.float32)
        reward = np.random.randn(self.reward_dim).astype(np.float32)
        done = False
        info = {}
        return next_obs, reward, done, info
