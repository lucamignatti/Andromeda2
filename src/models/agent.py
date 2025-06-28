"""
Hierarchical Agent combining the Strategic Planner and Motor Controller.
This is the main Andromeda2 agent that implements the brain-muscle architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import warnings

from .planner import StrategicPlanner, PlannerWithMemoryReplay, create_planner
from .controller import MotorController, create_controller


class Andromeda2Agent(nn.Module):
    """
    Hierarchical Reinforcement Learning Agent for Rocket League.

    Combines a strategic planner (xLSTM-based "brain") with a motor controller
    (MLP-based "muscles") to achieve both strategic thinking and mechanical precision.
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int = 8,
        goal_vector_dim: int = 12,
        planner_config: Optional[Dict[str, Any]] = None,
        controller_config: Optional[Dict[str, Any]] = None,
        planner_update_freq: int = 8,
        use_memory_replay: bool = False,
        controller_type: str = 'basic',
        goal_vector_scaling: float = 1.0,
        training_mode: str = 'hierarchical'  # 'hierarchical', 'planner_only', 'controller_only'
    ):
        """
        Initialize Andromeda2 Agent.

        Args:
            observation_size: Size of game state observations
            action_size: Size of action space
            goal_vector_dim: Dimension of goal vectors
            planner_config: Configuration for strategic planner
            controller_config: Configuration for motor controller
            planner_update_freq: How often planner updates (in steps)
            use_memory_replay: Whether to use memory replay for planner
            controller_type: Type of controller to use
            goal_vector_scaling: Scaling factor for goal vectors
            training_mode: Training mode ('hierarchical', 'planner_only', 'controller_only')
        """
        super(Andromeda2Agent, self).__init__()

        self.observation_size = observation_size
        self.action_size = action_size
        self.goal_vector_dim = goal_vector_dim
        self.planner_update_freq = planner_update_freq
        self.goal_vector_scaling = goal_vector_scaling
        self.training_mode = training_mode

        # Default configurations
        if planner_config is None:
            planner_config = {
                'hidden_size': 512,
                'num_layers': 3,
                'slstm_ratio': 0.7,
                'dropout': 0.1
            }

        if controller_config is None:
            controller_config = {
                'hidden_sizes': [512, 512, 256, 128],
                'dropout': 0.1,
                'use_attention': False,
                'use_goal_conditioning': 'concat'
            }

        # Create strategic planner using factory function
        planner_type = 'memory_replay' if use_memory_replay else 'basic'
        self.planner = create_planner(
            planner_type=planner_type,
            observation_size=observation_size,
            goal_vector_dim=goal_vector_dim,
            **planner_config
        )

        # Create motor controller
        self.controller = create_controller(
            controller_type=controller_type,
            observation_size=observation_size,
            goal_vector_dim=goal_vector_dim,
            action_size=action_size,
            **controller_config
        )

        # State tracking
        self.step_count = 0
        self.current_goal_vector = None
        self.planner_state = None

        # Episode statistics
        self.episode_stats = {
            'planner_updates': 0,
            'controller_steps': 0,
            'goal_vector_changes': 0
        }

    def reset_episode(self, batch_size: int = 1, device: torch.device = None):
        """Reset agent state for new episode."""
        if device is None:
            device = next(self.parameters()).device

        self.step_count = 0
        self.current_goal_vector = torch.zeros(batch_size, self.goal_vector_dim, device=device)
        self.planner_state = self.planner.reset_state(batch_size)

        # Reset episode statistics
        self.episode_stats = {
            'planner_updates': 0,
            'controller_steps': 0,
            'goal_vector_changes': 0
        }

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical agent.

        Args:
            observations: Game state observations
            deterministic: Whether to use deterministic actions
            return_components: Whether to return individual component outputs

        Returns:
            Dictionary containing actions, values, and optionally component outputs
        """
        batch_size = observations.size(0)
        device = observations.device

        # Initialize if needed
        if self.current_goal_vector is None or self.current_goal_vector.size(0) != batch_size:
            self.reset_episode(batch_size, device)

        # Update planner if it's time
        planner_output = None
        if self.step_count % self.planner_update_freq == 0 or self.training_mode == 'planner_only':
            planner_output = self._update_planner(observations)

        # Get controller output
        controller_output = self._get_controller_output(observations, deterministic)

        # Combine outputs
        result = {
            'actions': controller_output['actions'],
            'planner_values': planner_output['values'] if planner_output else torch.zeros(batch_size, device=device),
            'controller_values': controller_output['values'],
            'goal_vectors': self.current_goal_vector,
        }

        # Add component outputs if requested
        if return_components:
            result['planner_output'] = planner_output
            result['controller_output'] = controller_output

        self.step_count += 1
        self.episode_stats['controller_steps'] += 1

        return result

    def _update_planner(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Update strategic planner and goal vector."""
        if self.training_mode == 'controller_only':
            # Don't update planner in controller-only mode
            batch_size = observations.size(0)
            device = observations.device
            return {
                'goal_vectors': self.current_goal_vector,
                'values': torch.zeros(batch_size, device=device)
            }

        # Add sequence dimension if needed
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)  # (1, batch_size, obs_size)

        # Get planner output
        planner_output = self.planner(
            observations,
            self.planner_state,
            return_state=True
        )

        # Update state
        self.planner_state = planner_output.get('state')

        # Update goal vector with scaling
        new_goal_vector = planner_output['goal_vectors'] * self.goal_vector_scaling

        # Check if goal vector changed significantly
        if self.current_goal_vector is not None:
            goal_change = torch.norm(new_goal_vector - self.current_goal_vector, dim=-1).mean()
            if goal_change > 0.1:  # Threshold for significant change
                self.episode_stats['goal_vector_changes'] += 1

        self.current_goal_vector = new_goal_vector
        self.episode_stats['planner_updates'] += 1

        return planner_output

    def _get_controller_output(self, observations: torch.Tensor, deterministic: bool) -> Dict[str, torch.Tensor]:
        """Get motor controller output."""
        if self.training_mode == 'planner_only':
            # Return dummy actions in planner-only mode
            batch_size = observations.size(0)
            device = observations.device
            return {
                'actions': torch.zeros(batch_size, self.action_size, device=device),
                'values': torch.zeros(batch_size, device=device)
            }

        return self.controller(observations, self.current_goal_vector, deterministic)

    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Get action for a single observation.

        Args:
            observation: Single game state observation
            deterministic: Whether to use deterministic action

        Returns:
            Action tensor
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        with torch.no_grad():
            result = self.forward(observation, deterministic)
            return result['actions'].squeeze(0)

    def get_value(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get value estimates from both planner and controller.

        Args:
            observation: Game state observation

        Returns:
            Tuple of (planner_value, controller_value)
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        with torch.no_grad():
            result = self.forward(observation)
            return result['planner_values'], result['controller_values']

    def get_goal_vector(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Get current goal vector for given observation.

        Args:
            observation: Game state observation

        Returns:
            Goal vector tensor
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        with torch.no_grad():
            self.forward(observation)  # This will update goal vector if needed
            return self.current_goal_vector.squeeze(0) if self.current_goal_vector.size(0) == 1 else self.current_goal_vector

    def set_training_mode(self, mode: str):
        """
        Set training mode for the agent.

        Args:
            mode: Training mode ('hierarchical', 'planner_only', 'controller_only')
        """
        valid_modes = ['hierarchical', 'planner_only', 'controller_only']
        if mode not in valid_modes:
            raise ValueError(f"Invalid training mode: {mode}. Must be one of {valid_modes}")

        self.training_mode = mode

        # Freeze/unfreeze parameters based on mode
        if mode == 'planner_only':
            # Freeze controller parameters
            for param in self.controller.parameters():
                param.requires_grad = False
        elif mode == 'controller_only':
            # Freeze planner parameters
            for param in self.planner.parameters():
                param.requires_grad = False
        else:  # hierarchical
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        return self.episode_stats.copy()

    def save_checkpoint(self, filepath: str, include_stats: bool = True):
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
            include_stats: Whether to include episode statistics
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'observation_size': self.observation_size,
            'action_size': self.action_size,
            'goal_vector_dim': self.goal_vector_dim,
            'planner_update_freq': self.planner_update_freq,
            'goal_vector_scaling': self.goal_vector_scaling,
            'training_mode': self.training_mode
        }

        if include_stats:
            checkpoint['episode_stats'] = self.episode_stats

        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str,
        planner_config: Optional[Dict[str, Any]] = None,
        controller_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Load agent from checkpoint.

        Args:
            filepath: Path to checkpoint file
            planner_config: Override planner config
            controller_config: Override controller config
            **kwargs: Additional arguments

        Returns:
            Loaded agent instance
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        # Create agent with saved configuration
        agent = cls(
            observation_size=checkpoint['observation_size'],
            action_size=checkpoint['action_size'],
            goal_vector_dim=checkpoint['goal_vector_dim'],
            planner_update_freq=checkpoint['planner_update_freq'],
            goal_vector_scaling=checkpoint['goal_vector_scaling'],
            training_mode=checkpoint['training_mode'],
            planner_config=planner_config,
            controller_config=controller_config,
            **kwargs
        )

        # Load state dict
        agent.load_state_dict(checkpoint['model_state_dict'])

        # Load episode stats if available
        if 'episode_stats' in checkpoint:
            agent.episode_stats = checkpoint['episode_stats']

        return agent

    def analyze_goal_vectors(self, observations: torch.Tensor, num_steps: int = 10) -> Dict[str, Any]:
        """
        Analyze goal vector evolution over multiple steps.

        Args:
            observations: Sequence of observations
            num_steps: Number of steps to analyze

        Returns:
            Analysis results
        """
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)

        goal_vectors = []
        values = []

        with torch.no_grad():
            for i in range(min(num_steps, observations.size(0))):
                obs = observations[i:i+1]
                result = self.forward(obs)
                goal_vectors.append(result['goal_vectors'].cpu().numpy())
                values.append(result['planner_values'].cpu().numpy())

        goal_vectors = np.concatenate(goal_vectors, axis=0)
        values = np.concatenate(values, axis=0)

        # Calculate statistics
        analysis = {
            'goal_vector_mean': np.mean(goal_vectors, axis=0),
            'goal_vector_std': np.std(goal_vectors, axis=0),
            'goal_vector_range': np.ptp(goal_vectors, axis=0),
            'value_mean': np.mean(values),
            'value_std': np.std(values),
            'goal_vector_changes': np.sum(np.diff(goal_vectors, axis=0) ** 2, axis=1),
            'avg_goal_change': np.mean(np.sum(np.diff(goal_vectors, axis=0) ** 2, axis=1))
        }

        return analysis


class Andromeda2EnsembleAgent(nn.Module):
    """
    Ensemble version of Andromeda2 agent for improved robustness.
    """

    def __init__(
        self,
        num_agents: int,
        observation_size: int,
        action_size: int = 8,
        **agent_kwargs
    ):
        super().__init__()

        self.num_agents = num_agents
        self.agents = nn.ModuleList([
            Andromeda2Agent(
                observation_size=observation_size,
                action_size=action_size,
                **agent_kwargs
            )
            for _ in range(num_agents)
        ])

        # Ensemble weighting
        self.ensemble_weights = nn.Parameter(torch.ones(num_agents) / num_agents)

    def forward(self, observations: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble."""
        agent_outputs = []
        for agent in self.agents:
            output = agent(observations, deterministic)
            agent_outputs.append(output)

        # Weighted combination
        weights = F.softmax(self.ensemble_weights, dim=0)

        combined_actions = torch.zeros_like(agent_outputs[0]['actions'])
        combined_planner_values = torch.zeros_like(agent_outputs[0]['planner_values'])
        combined_controller_values = torch.zeros_like(agent_outputs[0]['controller_values'])

        for i, output in enumerate(agent_outputs):
            w = weights[i]
            combined_actions += w * output['actions']
            combined_planner_values += w * output['planner_values']
            combined_controller_values += w * output['controller_values']

        return {
            'actions': combined_actions,
            'planner_values': combined_planner_values,
            'controller_values': combined_controller_values,
            'ensemble_weights': weights,
            'individual_outputs': agent_outputs
        }

    def reset_episode(self, batch_size: int = 1, device: torch.device = None):
        """Reset all agents."""
        for agent in self.agents:
            agent.reset_episode(batch_size, device)
