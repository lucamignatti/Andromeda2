"""
MLP-based Controller for real-time action execution in Rocket League.
This module implements the "muscles" of the Andromeda2 agent using a fast MLP architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import math


class MLPBlock(nn.Module):
    """
    Basic MLP block with residual connections and normalization.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        super(MLPBlock, self).__init__()

        self.use_residual = use_residual and (input_size == output_size)

        # Main layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

        # Normalization
        if use_batch_norm:
            self.norm1 = nn.BatchNorm1d(hidden_size)
            self.norm2 = nn.BatchNorm1d(output_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(output_size)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP block."""
        residual = x if self.use_residual else None

        # First layer
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Second layer
        x = self.linear2(x)
        x = self.norm2(x)

        # Residual connection
        if residual is not None:
            x = x + residual

        x = self.activation(x)

        return x


class AttentionBlock(nn.Module):
    """
    Simple attention mechanism for focusing on relevant parts of the input.
    """

    def __init__(self, input_size: int, attention_size: int = 64):
        super(AttentionBlock, self).__init__()

        self.attention_size = attention_size

        # Attention weights
        self.query = nn.Linear(input_size, attention_size)
        self.key = nn.Linear(input_size, attention_size)
        self.value = nn.Linear(input_size, input_size)

        self.scale = math.sqrt(attention_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to input."""
        batch_size, seq_len = x.size(0), 1

        # If input is 2D, add sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_size)

        Q = self.query(x)  # (batch_size, seq_len, attention_size)
        K = self.key(x)    # (batch_size, seq_len, attention_size)
        V = self.value(x)  # (batch_size, seq_len, input_size)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention
        attended = torch.matmul(attention_weights, V)

        # Remove sequence dimension if it was added
        if attended.size(1) == 1:
            attended = attended.squeeze(1)

        return attended


class MotorController(nn.Module):
    """
    Motor Controller using MLP architecture for real-time action execution.
    The "muscles" of the Andromeda2 agent responsible for translating strategic intent into precise actions.
    """

    def __init__(
        self,
        observation_size: int,
        goal_vector_dim: int = 12,
        action_size: int = 8,  # Standard Rocket League action space
        hidden_sizes: List[int] = [512, 512, 256, 128],
        dropout: float = 0.1,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        use_attention: bool = False,
        use_goal_conditioning: str = 'concat'  # 'concat', 'film', 'cross_attention'
    ):
        """
        Initialize Motor Controller.

        Args:
            observation_size: Size of game state observations
            goal_vector_dim: Dimension of goal vector from planner
            action_size: Size of action space
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
            activation: Activation function type
            use_batch_norm: Whether to use batch normalization
            use_attention: Whether to use attention mechanism
            use_goal_conditioning: How to condition on goal vector
        """
        super(MotorController, self).__init__()

        self.observation_size = observation_size
        self.goal_vector_dim = goal_vector_dim
        self.action_size = action_size
        self.use_attention = use_attention
        self.use_goal_conditioning = use_goal_conditioning

        # Input processing
        if use_goal_conditioning == 'concat':
            input_size = observation_size + goal_vector_dim
        else:
            input_size = observation_size

        # Goal vector processing (for FiLM conditioning)
        if use_goal_conditioning == 'film':
            self.goal_processor = nn.ModuleList([
                nn.Linear(goal_vector_dim, hidden_size * 2)  # For scale and shift
                for hidden_size in hidden_sizes
            ])
        elif use_goal_conditioning == 'cross_attention':
            self.goal_attention = AttentionBlock(observation_size + goal_vector_dim)

        # Attention mechanism
        if use_attention:
            self.attention = AttentionBlock(input_size)

        # Main MLP layers
        layer_sizes = [input_size] + hidden_sizes
        self.mlp_layers = nn.ModuleList([
            MLPBlock(
                input_size=layer_sizes[i],
                hidden_size=layer_sizes[i+1],
                output_size=layer_sizes[i+1],
                dropout=dropout,
                activation=activation,
                use_batch_norm=use_batch_norm,
                use_residual=True if i > 0 else False
            )
            for i in range(len(layer_sizes) - 1)
        ])

        # Action heads
        final_size = hidden_sizes[-1]

        # Continuous actions (throttle, steer, pitch, yaw, roll)
        self.continuous_head = nn.Sequential(
            nn.Linear(final_size, final_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(final_size // 2, 5),  # throttle, steer, pitch, yaw, roll
            nn.Tanh()  # Continuous actions in [-1, 1]
        )

        # Discrete actions (jump, boost, handbrake)
        self.discrete_head = nn.Sequential(
            nn.Linear(final_size, final_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(final_size // 2, 3),  # jump, boost, handbrake
            nn.Sigmoid()  # Discrete actions as probabilities
        )

        # Value head for intrinsic reward prediction
        self.value_head = nn.Sequential(
            nn.Linear(final_size, final_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(final_size // 2, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        observations: torch.Tensor,
        goal_vectors: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through motor controller.

        Args:
            observations: Game state observations
            goal_vectors: Goal vectors from strategic planner
            deterministic: Whether to use deterministic actions

        Returns:
            Dictionary containing actions and values
        """
        batch_size = observations.size(0)

        # Process inputs based on conditioning method
        if self.use_goal_conditioning == 'concat':
            x = torch.cat([observations, goal_vectors], dim=-1)
        elif self.use_goal_conditioning == 'cross_attention':
            combined = torch.cat([observations, goal_vectors], dim=-1)
            x = self.goal_attention(combined)
        else:
            x = observations

        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)

        # Pass through MLP layers
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)

            # Apply FiLM conditioning if enabled
            if self.use_goal_conditioning == 'film' and i < len(self.goal_processor):
                goal_params = self.goal_processor[i](goal_vectors)
                scale, shift = goal_params.chunk(2, dim=-1)
                x = x * (1 + scale) + shift

        # Generate actions
        continuous_actions = self.continuous_head(x)
        discrete_logits = self.discrete_head(x)

        # Sample discrete actions
        if deterministic:
            discrete_actions = (discrete_logits > 0.5).float()
        else:
            discrete_actions = torch.bernoulli(discrete_logits)

        # Combine actions
        actions = torch.cat([continuous_actions, discrete_actions], dim=-1)

        # Generate value estimate
        values = self.value_head(x).squeeze(-1)

        return {
            'actions': actions,
            'continuous_actions': continuous_actions,
            'discrete_actions': discrete_actions,
            'discrete_logits': discrete_logits,
            'values': values
        }

    def get_action(
        self,
        observation: torch.Tensor,
        goal_vector: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Get action for single observation.

        Args:
            observation: Single observation
            goal_vector: Goal vector from planner
            deterministic: Whether to use deterministic actions

        Returns:
            Action tensor
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if goal_vector.dim() == 1:
            goal_vector = goal_vector.unsqueeze(0)

        with torch.no_grad():
            result = self.forward(observation, goal_vector, deterministic)
            return result['actions'].squeeze(0)


class AdaptiveController(MotorController):
    """
    Adaptive Motor Controller that can adjust its behavior based on performance feedback.
    """

    def __init__(self, *args, adaptation_rate: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)

        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.adaptation_weights = nn.Parameter(torch.ones(len(self.mlp_layers)))

    def update_performance(self, performance_score: float):
        """Update performance history for adaptation."""
        self.performance_history.append(performance_score)
        if len(self.performance_history) > 100:  # Keep last 100 scores
            self.performance_history.pop(0)

    def forward(self, observations: torch.Tensor, goal_vectors: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive layer weighting."""
        batch_size = observations.size(0)

        # Process inputs
        if self.use_goal_conditioning == 'concat':
            x = torch.cat([observations, goal_vectors], dim=-1)
        elif self.use_goal_conditioning == 'cross_attention':
            combined = torch.cat([observations, goal_vectors], dim=-1)
            x = self.goal_attention(combined)
        else:
            x = observations

        if self.use_attention:
            x = self.attention(x)

        # Pass through MLP layers with adaptive weighting
        for i, layer in enumerate(self.mlp_layers):
            layer_output = layer(x)

            # Apply adaptive weighting
            weight = torch.sigmoid(self.adaptation_weights[i])
            x = weight * layer_output + (1 - weight) * x

            # Apply FiLM conditioning if enabled
            if self.use_goal_conditioning == 'film' and i < len(self.goal_processor):
                goal_params = self.goal_processor[i](goal_vectors)
                scale, shift = goal_params.chunk(2, dim=-1)
                x = x * (1 + scale) + shift

        # Generate actions
        continuous_actions = self.continuous_head(x)
        discrete_logits = self.discrete_head(x)

        if deterministic:
            discrete_actions = (discrete_logits > 0.5).float()
        else:
            discrete_actions = torch.bernoulli(discrete_logits)

        actions = torch.cat([continuous_actions, discrete_actions], dim=-1)
        values = self.value_head(x).squeeze(-1)

        return {
            'actions': actions,
            'continuous_actions': continuous_actions,
            'discrete_actions': discrete_actions,
            'discrete_logits': discrete_logits,
            'values': values,
            'adaptation_weights': self.adaptation_weights
        }


class EnsembleController(nn.Module):
    """
    Ensemble of multiple controllers for robust action selection.
    """

    def __init__(
        self,
        observation_size: int,
        goal_vector_dim: int = 12,
        action_size: int = 8,
        num_controllers: int = 3,
        **controller_kwargs
    ):
        super().__init__()

        self.num_controllers = num_controllers

        # Create ensemble of controllers
        self.controllers = nn.ModuleList([
            MotorController(
                observation_size=observation_size,
                goal_vector_dim=goal_vector_dim,
                action_size=action_size,
                **controller_kwargs
            )
            for _ in range(num_controllers)
        ])

        # Ensemble weighting network
        self.ensemble_weights = nn.Sequential(
            nn.Linear(observation_size + goal_vector_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_controllers),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        observations: torch.Tensor,
        goal_vectors: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble."""
        batch_size = observations.size(0)

        # Get outputs from all controllers
        controller_outputs = []
        for controller in self.controllers:
            output = controller(observations, goal_vectors, deterministic)
            controller_outputs.append(output)

        # Compute ensemble weights
        combined_input = torch.cat([observations, goal_vectors], dim=-1)
        weights = self.ensemble_weights(combined_input)  # (batch_size, num_controllers)

        # Weighted combination of outputs
        actions = torch.zeros_like(controller_outputs[0]['actions'])
        values = torch.zeros_like(controller_outputs[0]['values'])

        for i, output in enumerate(controller_outputs):
            weight = weights[:, i:i+1]
            actions += weight * output['actions']
            values += weights[:, i] * output['values']

        return {
            'actions': actions,
            'values': values,
            'ensemble_weights': weights,
            'individual_outputs': controller_outputs
        }


# Factory function for creating controllers
def create_controller(
    controller_type: str,
    observation_size: int,
    goal_vector_dim: int = 12,
    action_size: int = 8,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of controllers.

    Args:
        controller_type: Type of controller ('basic', 'adaptive', 'ensemble')
        observation_size: Size of observations
        goal_vector_dim: Dimension of goal vectors
        action_size: Size of action space
        **kwargs: Additional arguments

    Returns:
        Controller instance
    """
    if controller_type == 'basic':
        return MotorController(
            observation_size=observation_size,
            goal_vector_dim=goal_vector_dim,
            action_size=action_size,
            **kwargs
        )
    elif controller_type == 'adaptive':
        return AdaptiveController(
            observation_size=observation_size,
            goal_vector_dim=goal_vector_dim,
            action_size=action_size,
            **kwargs
        )
    elif controller_type == 'ensemble':
        return EnsembleController(
            observation_size=observation_size,
            goal_vector_dim=goal_vector_dim,
            action_size=action_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
