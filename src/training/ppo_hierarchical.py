"""
Hierarchical Proximal Policy Optimization (PPO) for Andromeda2 Agent.
Implements dual-reward training system with separate optimization for planner and controller.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import wandb
from collections import deque
import warnings

from ..models.agent import Andromeda2Agent
from ..environments.vectorized import VectorizedRLGymEnv
from ..utils.memory import HierarchicalRolloutBuffer
from ..utils.metrics import TrainingMetrics


class HierarchicalPPOTrainer:
    """
    Hierarchical PPO trainer for Andromeda2 agent.

    Implements the dual-reward system:
    - Planner receives extrinsic rewards (game outcomes)
    - Controller receives intrinsic rewards (goal achievement)
    """

    def __init__(
        self,
        agent: Andromeda2Agent,
        env: VectorizedRLGymEnv,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        epochs_per_update: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        intrinsic_reward_weights: Optional[Dict[str, float]] = None,
        planner_lr_multiplier: float = 0.5,
        controller_lr_multiplier: float = 1.0,
        separate_optimizers: bool = True,
        normalize_advantages: bool = True,
        normalize_intrinsic_rewards: bool = True,
        target_kl: float = 0.02,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize hierarchical PPO trainer.

        Args:
            agent: Andromeda2 agent to train
            env: Vectorized environment
            learning_rate: Base learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping range
            entropy_coef: Entropy coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            epochs_per_update: Number of epochs per PPO update
            batch_size: Batch size for training
            n_steps: Number of steps per rollout
            intrinsic_reward_weights: Weights for intrinsic reward components
            planner_lr_multiplier: Learning rate multiplier for planner
            controller_lr_multiplier: Learning rate multiplier for controller
            separate_optimizers: Whether to use separate optimizers
            normalize_advantages: Whether to normalize advantages
            normalize_intrinsic_rewards: Whether to normalize intrinsic rewards
            target_kl: Target KL divergence for early stopping
            device: Device to use for training
        """
        self.agent = agent.to(device)
        self.env = env
        self.device = device

        # Training hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.normalize_advantages = normalize_advantages
        self.normalize_intrinsic_rewards = normalize_intrinsic_rewards
        self.target_kl = target_kl

        # Intrinsic reward weights
        if intrinsic_reward_weights is None:
            self.intrinsic_reward_weights = {
                'car_velocity': 1.0,
                'ball_velocity': 1.0,
                'car_to_ball_pos': 1.0,
                'ball_to_goal_pos': 1.0
            }
        else:
            self.intrinsic_reward_weights = intrinsic_reward_weights

        # Optimizers
        if separate_optimizers:
            self.planner_optimizer = optim.Adam(
                self.agent.planner.parameters(),
                lr=learning_rate * planner_lr_multiplier,
                eps=1e-5
            )
            self.controller_optimizer = optim.Adam(
                self.agent.controller.parameters(),
                lr=learning_rate * controller_lr_multiplier,
                eps=1e-5
            )
        else:
            self.optimizer = optim.Adam(
                self.agent.parameters(),
                lr=learning_rate,
                eps=1e-5
            )
            self.planner_optimizer = self.optimizer
            self.controller_optimizer = self.optimizer

        # Rollout buffer
        self.rollout_buffer = HierarchicalRolloutBuffer(
            buffer_size=n_steps,
            observation_space=env.observation_space,
            action_space=env.action_space,
            goal_vector_dim=agent.goal_vector_dim,
            n_envs=env.num_envs,
            gae_lambda=gae_lambda,
            gamma=gamma,
            device=device
        )

        # Metrics tracking
        self.metrics = TrainingMetrics()

        # Running statistics for normalization
        self.intrinsic_reward_stats = {
            'mean': 0.0,
            'var': 1.0,
            'count': 0
        }

        # Training state
        self.num_timesteps = 0
        self.num_updates = 0
        self._last_obs = None
        self._last_episode_starts = None

    def collect_rollouts(self) -> bool:
        """
        Collect rollouts for training.

        Returns:
            True if collection was successful
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Clear buffer
        self.rollout_buffer.reset()

        self.agent.eval()

        for step in range(self.n_steps):
            with torch.no_grad():
                # Get agent output
                agent_output = self.agent(
                    self._last_obs,
                    deterministic=False,
                    return_components=True
                )

                actions = agent_output['actions']
                planner_values = agent_output['planner_values']
                controller_values = agent_output['controller_values']
                goal_vectors = agent_output['goal_vectors']

                # Convert actions to numpy for environment
                actions_np = actions.cpu().numpy()

            # Step environment
            new_obs, rewards, dones, infos = self.env.step(actions_np)

            # Process rewards (extrinsic and intrinsic)
            extrinsic_rewards, intrinsic_rewards = self._process_rewards(
                rewards, infos, self._last_obs, new_obs, goal_vectors
            )

            # Store in buffer
            self.rollout_buffer.add(
                obs=self._last_obs,
                actions=actions,
                rewards_extrinsic=extrinsic_rewards,
                rewards_intrinsic=intrinsic_rewards,
                episode_starts=self._last_episode_starts,
                values_planner=planner_values,
                values_controller=controller_values,
                log_probs=self._get_action_log_probs(actions, agent_output),
                goal_vectors=goal_vectors
            )

            # Update observations
            self._last_obs = torch.FloatTensor(new_obs).to(self.device)
            self._last_episode_starts = torch.FloatTensor(dones).to(self.device)

            self.num_timesteps += self.env.num_envs

            # Handle episode resets
            for i, done in enumerate(dones):
                if done:
                    self.agent.reset_episode(1, self.device)

        # Compute final values for GAE
        with torch.no_grad():
            final_output = self.agent(self._last_obs)
            final_planner_values = final_output['planner_values']
            final_controller_values = final_output['controller_values']

        self.rollout_buffer.compute_returns_and_advantage(
            last_values_planner=final_planner_values,
            last_values_controller=final_controller_values,
            dones=self._last_episode_starts
        )

        return True

    def _process_rewards(
        self,
        env_rewards: np.ndarray,
        infos: List[Dict],
        obs: torch.Tensor,
        new_obs: torch.Tensor,
        goal_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process environment rewards into extrinsic and intrinsic components.

        Args:
            env_rewards: Raw environment rewards
            infos: Environment info dictionaries
            obs: Previous observations
            new_obs: New observations
            goal_vectors: Current goal vectors

        Returns:
            Tuple of (extrinsic_rewards, intrinsic_rewards)
        """
        batch_size = len(env_rewards)

        # Extrinsic rewards (for planner) - direct from environment
        extrinsic_rewards = torch.FloatTensor(env_rewards).to(self.device)

        # Intrinsic rewards (for controller) - based on goal achievement
        intrinsic_rewards = self._calculate_intrinsic_rewards(
            obs, new_obs, goal_vectors
        )

        # Normalize intrinsic rewards if enabled
        if self.normalize_intrinsic_rewards:
            intrinsic_rewards = self._normalize_intrinsic_rewards(intrinsic_rewards)

        return extrinsic_rewards, intrinsic_rewards

    def _calculate_intrinsic_rewards(
        self,
        obs: torch.Tensor,
        new_obs: torch.Tensor,
        goal_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate intrinsic rewards based on goal vector achievement.

        This is a simplified version - you'll need to adapt based on your
        specific observation format from AdvancedObs.
        """
        batch_size = obs.size(0)
        intrinsic_rewards = torch.zeros(batch_size, device=self.device)

        # Parse goal vector (12D: car_vel(3) + ball_vel(3) + car_to_ball(3) + ball_to_goal(3))
        target_car_vel = goal_vectors[:, :3]
        target_ball_vel = goal_vectors[:, 3:6]
        target_car_to_ball = goal_vectors[:, 6:9]
        target_ball_to_goal = goal_vectors[:, 9:12]

        # Extract state information from observations
        # NOTE: This is a placeholder - you need to implement based on AdvancedObs format
        # For now, assuming a simplified observation format

        # You would extract these from the actual observation:
        # actual_car_vel = extract_car_velocity(new_obs)
        # actual_ball_vel = extract_ball_velocity(new_obs)
        # actual_car_to_ball = extract_car_to_ball_position(new_obs)
        # actual_ball_to_goal = extract_ball_to_goal_position(new_obs)

        # Placeholder calculations (replace with actual state extraction)
        for i in range(batch_size):
            # Extract actual state (this is where you'd parse AdvancedObs)
            actual_car_vel = torch.zeros(3, device=self.device)  # Replace
            actual_ball_vel = torch.zeros(3, device=self.device)  # Replace
            actual_car_to_ball = torch.zeros(3, device=self.device)  # Replace
            actual_ball_to_goal = torch.zeros(3, device=self.device)  # Replace

            # Calculate negative squared errors (closer to target = higher reward)
            car_vel_error = -torch.sum((actual_car_vel - target_car_vel[i]) ** 2)
            ball_vel_error = -torch.sum((actual_ball_vel - target_ball_vel[i]) ** 2)
            car_to_ball_error = -torch.sum((actual_car_to_ball - target_car_to_ball[i]) ** 2)
            ball_to_goal_error = -torch.sum((actual_ball_to_goal - target_ball_to_goal[i]) ** 2)

            # Weighted combination
            intrinsic_reward = (
                self.intrinsic_reward_weights['car_velocity'] * car_vel_error +
                self.intrinsic_reward_weights['ball_velocity'] * ball_vel_error +
                self.intrinsic_reward_weights['car_to_ball_pos'] * car_to_ball_error +
                self.intrinsic_reward_weights['ball_to_goal_pos'] * ball_to_goal_error
            )

            intrinsic_rewards[i] = intrinsic_reward

        return intrinsic_rewards

    def _normalize_intrinsic_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize intrinsic rewards using running statistics."""
        # Update running statistics
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item()
        batch_count = rewards.numel()

        delta = batch_mean - self.intrinsic_reward_stats['mean']
        total_count = self.intrinsic_reward_stats['count'] + batch_count

        new_mean = self.intrinsic_reward_stats['mean'] + delta * batch_count / total_count

        new_var = (
            self.intrinsic_reward_stats['var'] * self.intrinsic_reward_stats['count'] +
            batch_var * batch_count +
            delta ** 2 * self.intrinsic_reward_stats['count'] * batch_count / total_count
        ) / total_count

        self.intrinsic_reward_stats['mean'] = new_mean
        self.intrinsic_reward_stats['var'] = max(new_var, 1e-8)
        self.intrinsic_reward_stats['count'] = total_count

        # Normalize
        normalized = (rewards - self.intrinsic_reward_stats['mean']) / np.sqrt(self.intrinsic_reward_stats['var'])
        return normalized

    def _get_action_log_probs(
        self,
        actions: torch.Tensor,
        agent_output: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate log probabilities of actions."""
        # This is a simplified version - you may need to adapt based on your action space
        if 'controller_output' in agent_output:
            controller_output = agent_output['controller_output']
            if 'discrete_logits' in controller_output:
                # For discrete actions
                discrete_logits = controller_output['discrete_logits']
                discrete_actions = actions[:, -3:]  # Last 3 actions are discrete
                log_probs_discrete = F.binary_cross_entropy_with_logits(
                    discrete_logits, discrete_actions, reduction='none'
                ).sum(dim=-1)
                return -log_probs_discrete  # Convert to log prob

        # Fallback - assume continuous actions with unit variance
        return torch.zeros(actions.size(0), device=self.device)

    def train(self) -> Dict[str, float]:
        """
        Perform one training update.

        Returns:
            Dictionary of training metrics
        """
        self.agent.train()

        # Get training data
        rollout_data = self.rollout_buffer.get()

        train_stats = {
            'planner_loss': 0.0,
            'controller_loss': 0.0,
            'planner_value_loss': 0.0,
            'controller_value_loss': 0.0,
            'policy_loss': 0.0,
            'entropy_loss': 0.0,
            'kl_divergence': 0.0,
            'clip_fraction': 0.0,
            'explained_variance_planner': 0.0,
            'explained_variance_controller': 0.0
        }

        # Multiple epochs of training
        for epoch in range(self.epochs_per_update):
            # Get batches
            for batch_data in self.rollout_buffer.get_batches(self.batch_size):
                # Train planner
                planner_stats = self._train_planner(batch_data)

                # Train controller
                controller_stats = self._train_controller(batch_data)

                # Accumulate stats
                for key in planner_stats:
                    train_stats[key] += planner_stats[key]
                for key in controller_stats:
                    train_stats[key] += controller_stats[key]

        # Average stats
        num_batches = self.epochs_per_update * (self.n_steps // self.batch_size)
        for key in train_stats:
            train_stats[key] /= num_batches

        self.num_updates += 1

        return train_stats

    def _train_planner(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train the strategic planner."""
        obs = batch_data['observations']
        returns_extrinsic = batch_data['returns_extrinsic']
        advantages_extrinsic = batch_data['advantages_extrinsic']
        old_values_planner = batch_data['values_planner']
        goal_vectors = batch_data['goal_vectors']

        # Normalize advantages
        if self.normalize_advantages:
            advantages_extrinsic = (advantages_extrinsic - advantages_extrinsic.mean()) / (advantages_extrinsic.std() + 1e-8)

        # Forward pass through planner
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)  # Add sequence dimension

        planner_output = self.agent.planner(obs)
        values_planner = planner_output['values']
        new_goal_vectors = planner_output['goal_vectors']

        # Value loss
        values_pred = values_planner.squeeze(0) if values_planner.dim() == 2 else values_planner
        value_loss = F.mse_loss(values_pred, returns_extrinsic)

        # Goal vector consistency loss (encourage stability)
        goal_consistency_loss = F.mse_loss(new_goal_vectors, goal_vectors.detach())

        # Total planner loss
        planner_loss = self.value_loss_coef * value_loss + 0.1 * goal_consistency_loss

        # Backward pass
        self.planner_optimizer.zero_grad()
        planner_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.planner.parameters(), self.max_grad_norm)
        self.planner_optimizer.step()

        # Calculate explained variance
        explained_var = 1 - (returns_extrinsic - values_pred).var() / returns_extrinsic.var()

        return {
            'planner_loss': planner_loss.item(),
            'planner_value_loss': value_loss.item(),
            'explained_variance_planner': explained_var.item()
        }

    def _train_controller(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train the motor controller."""
        obs = batch_data['observations']
        actions = batch_data['actions']
        returns_intrinsic = batch_data['returns_intrinsic']
        advantages_intrinsic = batch_data['advantages_intrinsic']
        old_log_probs = batch_data['log_probs']
        old_values_controller = batch_data['values_controller']
        goal_vectors = batch_data['goal_vectors']

        # Normalize advantages
        if self.normalize_advantages:
            advantages_intrinsic = (advantages_intrinsic - advantages_intrinsic.mean()) / (advantages_intrinsic.std() + 1e-8)

        # Forward pass through controller
        controller_output = self.agent.controller(obs, goal_vectors, deterministic=False)
        values_controller = controller_output['values']

        # Value loss
        value_loss = F.mse_loss(values_controller, returns_intrinsic)

        # Policy loss (simplified - you may need to adapt based on action space)
        # For now, using MSE between predicted and taken actions
        predicted_actions = controller_output['actions']
        policy_loss = F.mse_loss(predicted_actions, actions)

        # Entropy loss (encourage exploration)
        entropy_loss = 0.0  # Placeholder - implement based on your action distribution

        # Total controller loss
        controller_loss = (
            policy_loss +
            self.value_loss_coef * value_loss -
            self.entropy_coef * entropy_loss
        )

        # Backward pass
        self.controller_optimizer.zero_grad()
        controller_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.controller.parameters(), self.max_grad_norm)
        self.controller_optimizer.step()

        # Calculate explained variance
        explained_var = 1 - (returns_intrinsic - values_controller).var() / returns_intrinsic.var()

        return {
            'controller_loss': controller_loss.item(),
            'controller_value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss,
            'explained_variance_controller': explained_var.item()
        }

    def learn(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 1,
        eval_env = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "HierarchicalPPO",
        reset_num_timesteps: bool = True
    ) -> "HierarchicalPPOTrainer":
        """
        Main training loop.

        Args:
            total_timesteps: Total number of timesteps to train
            callback: Callback function
            log_interval: Log interval
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            tb_log_name: Tensorboard log name
            reset_num_timesteps: Whether to reset timestep counter

        Returns:
            Self
        """
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.num_updates = 0

        # Initialize environment
        self._last_obs = torch.FloatTensor(self.env.reset()).to(self.device)
        self._last_episode_starts = torch.ones(self.env.num_envs, device=self.device)

        # Training loop
        while self.num_timesteps < total_timesteps:
            # Collect rollouts
            continue_training = self.collect_rollouts()

            if continue_training is False:
                break

            # Train
            train_stats = self.train()

            # Logging
            if self.num_updates % log_interval == 0:
                self._log_training_stats(train_stats)

            # Evaluation
            if eval_freq > 0 and self.num_updates % eval_freq == 0:
                self._evaluate(eval_env, n_eval_episodes)

            # Callback
            if callback is not None:
                if callback(locals(), globals()) is False:
                    break

        return self

    def _log_training_stats(self, train_stats: Dict[str, float]):
        """Log training statistics."""
        print(f"Update {self.num_updates}, Timesteps {self.num_timesteps}")
        for key, value in train_stats.items():
            print(f"  {key}: {value:.4f}")

        # Log to wandb if available
        try:
            wandb.log({
                "update": self.num_updates,
                "timesteps": self.num_timesteps,
                **train_stats
            })
        except:
            pass

    def _evaluate(self, eval_env, n_episodes: int):
        """Evaluate agent performance."""
        if eval_env is None:
            return

        self.agent.eval()
        episode_rewards = []

        for _ in range(n_episodes):
            obs = eval_env.reset()
            total_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action = self.agent.get_action(obs_tensor, deterministic=True)
                    action_np = action.cpu().numpy()

                obs, reward, done, _ = eval_env.step(action_np)
                total_reward += reward

            episode_rewards.append(total_reward)

        mean_reward = np.mean(episode_rewards)
        print(f"Eval mean reward: {mean_reward:.2f}")

        try:
            wandb.log({"eval_mean_reward": mean_reward})
        except:
            pass

    def save(self, path: str):
        """Save trainer state."""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'planner_optimizer_state_dict': self.planner_optimizer.state_dict(),
            'controller_optimizer_state_dict': self.controller_optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
            'num_updates': self.num_updates,
            'intrinsic_reward_stats': self.intrinsic_reward_stats
        }, path)

    def load(self, path: str):
        """Load trainer state."""
        checkpoint = torch.load(path, map_location=self.device)

        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.planner_optimizer.load_state_dict(checkpoint['planner_optimizer_state_dict'])
        self.controller_optimizer.load_state_dict(checkpoint['controller_optimizer_state_dict'])
        self.num_timesteps = checkpoint['num_timesteps']
        self.num_updates = checkpoint['num_updates']
        self.intrinsic_reward_stats = checkpoint['intrinsic_reward_stats']
