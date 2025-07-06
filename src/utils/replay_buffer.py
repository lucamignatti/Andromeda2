import numpy as np
from collections import deque

class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling trajectories.
    """
    def __init__(self, capacity, sequence_length):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)

    def add(self, trajectory):
        """
        Adds a complete trajectory to the buffer.
        A trajectory is a list of (observation, action, reward, next_observation) tuples.
        """
        if len(trajectory) >= self.sequence_length:
            self.buffer.append(trajectory)

    def sample(self, batch_size):
        """
        Samples a batch of sequences from the buffer.
        """
        # Get a random batch of trajectories
        trajectory_indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        
        observations, actions, rewards, next_observations = [], [], [], []

        for idx in trajectory_indices:
            trajectory = self.buffer[idx]
            start_idx = np.random.randint(0, len(trajectory) - self.sequence_length + 1)
            
            sequence = trajectory[start_idx : start_idx + self.sequence_length]
            
            obs_seq, act_seq, rew_seq, next_obs_seq = zip(*sequence)
            
            observations.append(np.array(obs_seq))
            actions.append(np.array(act_seq))
            rewards.append(np.array(rew_seq))
            next_observations.append(np.array(next_obs_seq))

        return (
            np.array(observations),
            np.array(actions),
            np.array(rewards),
            np.array(next_observations)
        )

    def __len__(self):
        return len(self.buffer)
