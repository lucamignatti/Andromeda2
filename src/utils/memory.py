"""
Hierarchical Rollout Buffer for dual-reward system in Andromeda2.
Handles separate storage and processing of extrinsic and intrinsic rewards.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Generator
import warnings


class HierarchicalRolloutBuffer:
    """
    Rollout buffer for hierarchical RL with separate extrinsic and intrinsic rewards.

    Stores experiences with dual reward signals:
    - Extrinsic rewards for the planner (game outcomes)
    - Intrinsic rewards for the controller (goal achievement)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: Union[int, tuple],
        action_space: Union[int, tuple],
        goal_vector_dim: int,
        n_envs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu"
    ):
        """
        Initialize hierarchical rollout buffer.

        Args:
            buffer_size: Size of the buffer (number of steps)
            observation_space: Observation space dimension or shape
            action_space: Action space dimension or shape
            goal_vector_dim: Dimension of goal vectors
            n_envs: Number of parallel environments
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device to store tensors
        """
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Determine observation and action dimensions
        if isinstance(observation_space, int):
            self.obs_shape = (observation_space,)
        else:
            self.obs_shape = observation_space

        if isinstance(action_space, int):
            self.action_shape = (action_space,)
        else:
            # Handle different action space types
            import gymnasium as gym
            if hasattr(action_space, 'shape'):
                self.action_shape = action_space.shape
            elif hasattr(action_space, 'nvec'):  # MultiDiscrete
                self.action_shape = (len(action_space.nvec),)
            elif hasattr(action_space, 'n'):  # Discrete
                self.action_shape = (1,)
            else:
                # Fallback for other types
                self.action_shape = (8,)  # Default Rocket League action size

        self.goal_vector_dim = goal_vector_dim

        # Initialize storage tensors
        self._init_storage()

        # Buffer state
        self.pos = 0
        self.full = False
        self.generator_ready = False

    def _init_storage(self):
        """Initialize storage tensors."""
        # Observations
        self.observations = torch.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=torch.float32,
            device=self.device
        )

        # Actions
        self.actions = torch.zeros(
            (self.buffer_size, self.n_envs) + self.action_shape,
            dtype=torch.float32,
            device=self.device
        )

        # Goal vectors
        self.goal_vectors = torch.zeros(
            (self.buffer_size, self.n_envs, self.goal_vector_dim),
            dtype=torch.float32,
            device=self.device
        )

        # Rewards (dual system)
        self.rewards_extrinsic = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )

        self.rewards_intrinsic = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )

        # Values (dual system)
        self.values_planner = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )

        self.values_controller = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )

        # Log probabilities
        self.log_probs = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )

        # Episode starts
        self.episode_starts = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )

        # Returns and advantages (computed later)
        self.returns_extrinsic = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )

        self.returns_intrinsic = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )

        self.advantages_extrinsic = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )

        self.advantages_intrinsic = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards_extrinsic: torch.Tensor,
        rewards_intrinsic: torch.Tensor,
        episode_starts: torch.Tensor,
        values_planner: torch.Tensor,
        values_controller: torch.Tensor,
        log_probs: torch.Tensor,
        goal_vectors: torch.Tensor
    ):
        """
        Add experience to the buffer.

        Args:
            obs: Observations
            actions: Actions taken
            rewards_extrinsic: Extrinsic rewards (for planner)
            rewards_intrinsic: Intrinsic rewards (for controller)
            episode_starts: Episode start flags
            values_planner: Value estimates from planner
            values_controller: Value estimates from controller
            log_probs: Log probabilities of actions
            goal_vectors: Goal vectors from planner
        """
        if len(obs.shape) > len(self.observations.shape[2:]):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # Ensure tensors are on the correct device
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards_extrinsic = rewards_extrinsic.to(self.device)
        rewards_intrinsic = rewards_intrinsic.to(self.device)
        episode_starts = episode_starts.to(self.device)
        values_planner = values_planner.to(self.device)
        values_controller = values_controller.to(self.device)
        log_probs = log_probs.to(self.device)
        goal_vectors = goal_vectors.to(self.device)

        # Store in buffer
        self.observations[self.pos] = obs.clone()
        self.actions[self.pos] = actions.clone()
        self.rewards_extrinsic[self.pos] = rewards_extrinsic.clone()
        self.rewards_intrinsic[self.pos] = rewards_intrinsic.clone()
        self.episode_starts[self.pos] = episode_starts.clone()
        self.values_planner[self.pos] = values_planner.flatten()
        self.values_controller[self.pos] = values_controller.flatten()
        self.log_probs[self.pos] = log_probs.flatten()
        self.goal_vectors[self.pos] = goal_vectors.clone()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(
        self,
        last_values_planner: torch.Tensor,
        last_values_controller: torch.Tensor,
        dones: torch.Tensor
    ):
        """
        Compute returns and advantages using GAE for both reward systems.

        Args:
            last_values_planner: Final value estimates from planner
            last_values_controller: Final value estimates from controller
            dones: Episode done flags
        """
        # Ensure correct device and shape
        last_values_planner = last_values_planner.to(self.device).flatten()
        last_values_controller = last_values_controller.to(self.device).flatten()
        dones = dones.to(self.device).flatten()

        # Compute for extrinsic rewards (planner)
        self._compute_gae(
            rewards=self.rewards_extrinsic,
            values=self.values_planner,
            last_values=last_values_planner,
            dones=dones,
            returns_out=self.returns_extrinsic,
            advantages_out=self.advantages_extrinsic
        )

        # Compute for intrinsic rewards (controller)
        self._compute_gae(
            rewards=self.rewards_intrinsic,
            values=self.values_controller,
            last_values=last_values_controller,
            dones=dones,
            returns_out=self.returns_intrinsic,
            advantages_out=self.advantages_intrinsic
        )

        self.generator_ready = True

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        last_values: torch.Tensor,
        dones: torch.Tensor,
        returns_out: torch.Tensor,
        advantages_out: torch.Tensor
    ):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward tensor
            values: Value estimates
            last_values: Final value estimates
            dones: Episode done flags
            returns_out: Output tensor for returns
            advantages_out: Output tensor for advantages
        """
        last_gae_lam = 0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = values[step + 1]

            delta = (
                rewards[step] +
                self.gamma * next_values * next_non_terminal -
                values[step]
            )

            last_gae_lam = (
                delta +
                self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )

            advantages_out[step] = last_gae_lam

        # Returns = advantages + values
        returns_out[:] = advantages_out + values

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get all data from the buffer.

        Returns:
            Dictionary containing all buffer data
        """
        assert self.full, "Buffer not full"
        assert self.generator_ready, "GAE not computed"

        indices = np.arange(self.buffer_size * self.n_envs)

        return self._get_samples(indices)

    def get_batches(self, batch_size: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generate batches of experiences.

        Args:
            batch_size: Size of each batch

        Yields:
            Batches of experiences
        """
        assert self.full, "Buffer not full"
        assert self.generator_ready, "GAE not computed"

        indices = np.arange(self.buffer_size * self.n_envs)
        np.random.shuffle(indices)

        start_idx = 0
        while start_idx < len(indices):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield self._get_samples(batch_indices)
            start_idx += batch_size

    def _get_samples(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Get samples for given indices.

        Args:
            indices: Indices to sample

        Returns:
            Dictionary of sampled data
        """
        # Convert buffer indices to (step, env) indices
        env_indices = indices % self.n_envs
        step_indices = indices // self.n_envs

        return {
            'observations': self.observations[step_indices, env_indices],
            'actions': self.actions[step_indices, env_indices],
            'values_planner': self.values_planner[step_indices, env_indices],
            'values_controller': self.values_controller[step_indices, env_indices],
            'log_probs': self.log_probs[step_indices, env_indices],
            'advantages_extrinsic': self.advantages_extrinsic[step_indices, env_indices],
            'advantages_intrinsic': self.advantages_intrinsic[step_indices, env_indices],
            'returns_extrinsic': self.returns_extrinsic[step_indices, env_indices],
            'returns_intrinsic': self.returns_intrinsic[step_indices, env_indices],
            'goal_vectors': self.goal_vectors[step_indices, env_indices],
            'episode_starts': self.episode_starts[step_indices, env_indices]
        }

    def reset(self):
        """Reset the buffer."""
        self.pos = 0
        self.full = False
        self.generator_ready = False

    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer_size if self.full else self.pos

    def __len__(self) -> int:
        """Get current buffer size."""
        return self.size()


class ReplayBuffer:
    """
    Simple replay buffer for off-policy learning (if needed).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: Union[int, tuple],
        action_space: Union[int, tuple],
        goal_vector_dim: int,
        n_envs: int = 1,
        device: str = "cpu"
    ):
        """
        Initialize replay buffer.

        Args:
            buffer_size: Maximum buffer size
            observation_space: Observation space dimension or shape
            action_space: Action space dimension or shape
            goal_vector_dim: Dimension of goal vectors
            n_envs: Number of parallel environments
            device: Device to store tensors
        """
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = device

        # Determine dimensions
        if isinstance(observation_space, int):
            self.obs_shape = (observation_space,)
        else:
            self.obs_shape = observation_space

        if isinstance(action_space, int):
            self.action_shape = (action_space,)
        else:
            self.action_shape = action_space

        self.goal_vector_dim = goal_vector_dim

        # Initialize storage
        self._init_storage()

        # Buffer state
        self.pos = 0
        self.full = False

    def _init_storage(self):
        """Initialize storage tensors."""
        self.observations = torch.zeros(
            (self.buffer_size,) + self.obs_shape,
            dtype=torch.float32,
            device=self.device
        )

        self.next_observations = torch.zeros(
            (self.buffer_size,) + self.obs_shape,
            dtype=torch.float32,
            device=self.device
        )

        self.actions = torch.zeros(
            (self.buffer_size,) + self.action_shape,
            dtype=torch.float32,
            device=self.device
        )

        self.goal_vectors = torch.zeros(
            (self.buffer_size, self.goal_vector_dim),
            dtype=torch.float32,
            device=self.device
        )

        self.rewards_extrinsic = torch.zeros(
            (self.buffer_size,),
            dtype=torch.float32,
            device=self.device
        )

        self.rewards_intrinsic = torch.zeros(
            (self.buffer_size,),
            dtype=torch.float32,
            device=self.device
        )

        self.dones = torch.zeros(
            (self.buffer_size,),
            dtype=torch.float32,
            device=self.device
        )

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward_extrinsic: float,
        reward_intrinsic: float,
        done: bool,
        goal_vector: torch.Tensor
    ):
        """
        Add experience to replay buffer.

        Args:
            obs: Current observation
            next_obs: Next observation
            action: Action taken
            reward_extrinsic: Extrinsic reward
            reward_intrinsic: Intrinsic reward
            done: Episode done flag
            goal_vector: Goal vector
        """
        self.observations[self.pos] = obs.to(self.device)
        self.next_observations[self.pos] = next_obs.to(self.device)
        self.actions[self.pos] = action.to(self.device)
        self.rewards_extrinsic[self.pos] = reward_extrinsic
        self.rewards_intrinsic[self.pos] = reward_intrinsic
        self.dones[self.pos] = done
        self.goal_vectors[self.pos] = goal_vector.to(self.device)

        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch from replay buffer.

        Args:
            batch_size: Batch size

        Returns:
            Dictionary of sampled data
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_indices = np.random.randint(0, upper_bound, size=batch_size)

        return {
            'observations': self.observations[batch_indices],
            'next_observations': self.next_observations[batch_indices],
            'actions': self.actions[batch_indices],
            'rewards_extrinsic': self.rewards_extrinsic[batch_indices],
            'rewards_intrinsic': self.rewards_intrinsic[batch_indices],
            'dones': self.dones[batch_indices],
            'goal_vectors': self.goal_vectors[batch_indices]
        }

    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer_size if self.full else self.pos

    def __len__(self) -> int:
        """Get current buffer size."""
        return self.size()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer for hierarchical RL.
    """

    def __init__(self, *args, alpha: float = 0.6, beta: float = 0.4, **kwargs):
        """
        Initialize prioritized replay buffer.

        Args:
            alpha: Prioritization exponent
            beta: Importance sampling exponent
            *args: Parent class arguments
            **kwargs: Parent class keyword arguments
        """
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

        # Priority storage
        self.priorities = torch.zeros(
            self.buffer_size,
            dtype=torch.float32,
            device=self.device
        )

    def add(self, *args, **kwargs):
        """Add experience with maximum priority."""
        super().add(*args, **kwargs)

        # Set maximum priority for new experience
        if self.pos == 0 and self.full:
            self.priorities[self.buffer_size - 1] = self.max_priority
        else:
            self.priorities[self.pos - 1] = self.max_priority

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Sample batch with prioritized sampling.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (batch_data, indices, importance_weights)
        """
        upper_bound = self.buffer_size if self.full else self.pos

        # Calculate sampling probabilities
        priorities = self.priorities[:upper_bound] ** self.alpha
        probabilities = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(upper_bound, batch_size, p=probabilities.cpu().numpy())

        # Calculate importance weights
        weights = (upper_bound * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Get batch data
        batch_data = {
            'observations': self.observations[indices],
            'next_observations': self.next_observations[indices],
            'actions': self.actions[indices],
            'rewards_extrinsic': self.rewards_extrinsic[indices],
            'rewards_intrinsic': self.rewards_intrinsic[indices],
            'dones': self.dones[indices],
            'goal_vectors': self.goal_vectors[indices]
        }

        return batch_data, torch.tensor(indices, device=self.device), weights

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """
        Update priorities for given indices.

        Args:
            indices: Indices to update
            priorities: New priorities
        """
        self.priorities[indices] = priorities.to(self.device)
        self.max_priority = max(self.max_priority, priorities.max().item())
