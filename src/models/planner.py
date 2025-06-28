"""
LSTM-based Strategic Planner (temporarily replacing xLSTM to fix configuration issues).
This module implements the "brain" of the Andromeda2 agent using standard LSTM architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import math


class StrategicPlanner(nn.Module):
    """
    Strategic Planner using official xLSTM implementation.
    The "brain" of the Andromeda2 agent responsible for high-level strategic planning.
    """

    def __init__(
        self,
        observation_size: int,
        hidden_size: int = 512,
        num_layers: int = 3,
        goal_vector_dim: int = 12,
        slstm_at_layer: Optional[List[int]] = None,
        mlstm_at_layer: Optional[List[int]] = None,
        dropout: float = 0.1,
        embedding_dim: Optional[int] = None,
        add_post_blocks_norm: bool = True,
        bias: bool = False,
        context_length: int = 2048,
        tie_weights: bool = False,
        slstm_hidden_size_factor: float = 1.0,
        mlstm_memory_dim_factor: float = 1.0,
        mlstm_num_heads: int = 4,
        **kwargs
    ):
        """
        Initialize Strategic Planner with official xLSTM implementation.

        Args:
            observation_size: Size of input observations
            hidden_size: Hidden layer size for xLSTM blocks
            num_layers: Number of xLSTM layers
            goal_vector_dim: Dimension of output goal vector
            slstm_at_layer: List of layer indices to use sLSTM blocks (default: [0, 2, 4, ...])
            mlstm_at_layer: List of layer indices to use mLSTM blocks (default: [1, 3, 5, ...])
            dropout: Dropout probability
            embedding_dim: Embedding dimension (defaults to hidden_size)
            add_post_blocks_norm: Whether to add normalization after blocks
            bias: Whether to use bias in linear layers
            context_length: Maximum context length for memory
            tie_weights: Whether to tie input/output embeddings
            slstm_hidden_size_factor: Factor for sLSTM hidden size
            mlstm_memory_dim_factor: Factor for mLSTM memory dimension
            mlstm_num_heads: Number of heads for mLSTM attention
            **kwargs: Additional arguments
        """
        super(StrategicPlanner, self).__init__()

        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.goal_vector_dim = goal_vector_dim
        self.embedding_dim = embedding_dim or hidden_size
        self.context_length = context_length

        # Default layer assignments (alternating sLSTM and mLSTM)
        if slstm_at_layer is None:
            slstm_at_layer = [i for i in range(num_layers) if i % 2 == 0]
        if mlstm_at_layer is None:
            mlstm_at_layer = [i for i in range(num_layers) if i % 2 == 1]

        self.slstm_at_layer = slstm_at_layer
        self.mlstm_at_layer = mlstm_at_layer

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(observation_size, self.embedding_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # Create xLSTM configuration
        self.lstm_config = self._create_lstm_config(
            hidden_size=hidden_size,
            num_layers=num_layers,
            slstm_at_layer=slstm_at_layer,
            mlstm_at_layer=mlstm_at_layer,
            dropout=dropout,
            add_post_blocks_norm=add_post_blocks_norm,
            bias=bias,
            context_length=context_length,
            slstm_hidden_size_factor=slstm_hidden_size_factor,
            mlstm_memory_dim_factor=mlstm_memory_dim_factor,
            mlstm_num_heads=mlstm_num_heads
        )

        # Create LSTM layers (temporary replacement for xLSTM)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bias=bias
        )

        # Goal vector head
        self.goal_head = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_size // 2, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_size // 2, goal_vector_dim, bias=bias),
            nn.Tanh()  # Normalize goal vector to [-1, 1]
        )

        # Value head (for RL training)
        self.value_head = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_size // 2, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_size // 2, 1, bias=bias)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _create_lstm_config(
        self,
        num_layers: int,
        hidden_size: int,
        dropout: float,
        slstm_at_layer: List[int],
        mlstm_at_layer: List[int],
        add_post_blocks_norm: bool,
        bias: bool,
        context_length: int,
        slstm_hidden_size_factor: float,
        mlstm_memory_dim_factor: float,
        mlstm_num_heads: int
    ) -> Dict[str, Any]:
        """Create LSTM configuration (temporary replacement for xLSTM)."""

        return {
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'dropout': dropout,
            'context_length': context_length,
            'bias': bias
        }

    def _init_weights(self, module):
        """Initialize weights using best practices for xLSTM."""
        if isinstance(module, nn.Linear):
            # Use Xavier/Glorot initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            # Use normal initialization for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Any] = None,
        return_state: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through strategic planner.

        Args:
            observations: Input observations of shape (seq_len, batch_size, obs_size)
            state: Previous hidden state from xLSTM stack
            return_state: Whether to return updated state

        Returns:
            Dictionary containing goal_vectors, values, and optionally state
        """
        seq_len, batch_size, _ = observations.size()

        # Input embedding
        embedded = self.input_embedding(observations.view(-1, self.observation_size))
        embedded = embedded.view(seq_len, batch_size, self.embedding_dim)

        # Pass through LSTM
        # LSTM expects (batch_size, seq_len, embedding_dim)
        embedded_transposed = embedded.transpose(0, 1)  # (batch_size, seq_len, embedding_dim)

        if state is not None:
            lstm_output, new_state = self.lstm(embedded_transposed, state)
        else:
            lstm_output, new_state = self.lstm(embedded_transposed)

        # Use only the last timestep for output
        last_hidden = lstm_output[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Generate goal vector
        goal_vectors = self.goal_head(last_hidden)

        # Generate value estimate
        values = self.value_head(last_hidden).squeeze(-1)

        result = {
            'goal_vectors': goal_vectors,
            'values': values
        }

        if return_state:
            result['state'] = new_state

        return result

    def get_goal_vector(
        self,
        observations: torch.Tensor,
        state: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """
        Get goal vector for current observations.

        Args:
            observations: Current observations
            state: Previous hidden state

        Returns:
            Tuple of (goal_vector, new_state)
        """
        with torch.no_grad():
            if observations.dim() == 2:
                observations = observations.unsqueeze(0)  # Add sequence dimension

            result = self.forward(observations, state, return_state=True)
            return result['goal_vectors'], result['state']

    def reset_state(self, batch_size: int = 1) -> Any:
        """Reset/initialize hidden state for new episode."""
        # Initialize LSTM hidden state
        device = next(self.parameters()).device
        num_layers = self.lstm_config['num_layers']
        hidden_size = self.lstm_config['hidden_size']

        h0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        return (h0, c0)

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_usage(self, seq_len: int, batch_size: int) -> Dict[str, int]:
        """Estimate memory usage for given sequence length and batch size."""
        # This is an approximation
        param_memory = self.get_num_params() * 4  # 4 bytes per float32

        # Approximate activation memory (this is rough estimation)
        activation_memory = (
            seq_len * batch_size * self.embedding_dim *
            self.num_layers * 8  # Multiple activations per layer
        ) * 4  # 4 bytes per float32

        return {
            'parameters_bytes': param_memory,
            'activations_bytes': activation_memory,
            'total_bytes': param_memory + activation_memory,
            'parameters_mb': param_memory / (1024 * 1024),
            'activations_mb': activation_memory / (1024 * 1024),
            'total_mb': (param_memory + activation_memory) / (1024 * 1024)
        }


class PlannerWithMemoryReplay(StrategicPlanner):
    """
    Enhanced Strategic Planner with memory replay capabilities.
    Stores and replays important strategic decisions for better learning.
    """

    def __init__(self, *args, memory_capacity: int = 10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_capacity = memory_capacity
        self.strategic_memory = []
        self.memory_weights = []

    def store_strategic_decision(
        self,
        observation: torch.Tensor,
        goal_vector: torch.Tensor,
        reward: float,
        importance: float = 1.0
    ):
        """Store important strategic decisions in memory."""
        if len(self.strategic_memory) >= self.memory_capacity:
            # Remove least important memory
            min_idx = np.argmin(self.memory_weights)
            self.strategic_memory.pop(min_idx)
            self.memory_weights.pop(min_idx)

        self.strategic_memory.append({
            'observation': observation.detach().cpu(),
            'goal_vector': goal_vector.detach().cpu(),
            'reward': reward,
            'timestamp': len(self.strategic_memory)
        })
        self.memory_weights.append(importance)

    def sample_strategic_memories(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample strategic memories for replay training."""
        if len(self.strategic_memory) < batch_size:
            return None

        # Weighted sampling based on importance
        weights = np.array(self.memory_weights)
        weights = weights / weights.sum()

        indices = np.random.choice(
            len(self.strategic_memory),
            size=batch_size,
            p=weights,
            replace=False
        )

        batch = [self.strategic_memory[i] for i in indices]

        return {
            'observations': torch.stack([item['observation'] for item in batch]),
            'goal_vectors': torch.stack([item['goal_vector'] for item in batch]),
            'rewards': torch.tensor([item['reward'] for item in batch], dtype=torch.float),
            'timestamps': torch.tensor([item['timestamp'] for item in batch], dtype=torch.long)
        }


class AdaptivePlanner(StrategicPlanner):
    """
    Adaptive Strategic Planner that can adjust its planning frequency based on game state.
    """

    def __init__(
        self,
        *args,
        adaptation_threshold: float = 0.1,
        min_plan_frequency: int = 4,
        max_plan_frequency: int = 16,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.adaptation_threshold = adaptation_threshold
        self.min_plan_frequency = min_plan_frequency
        self.max_plan_frequency = max_plan_frequency
        self.last_goal_vector = None
        self.steps_since_update = 0

    def should_update_plan(self, current_state: torch.Tensor) -> bool:
        """Determine if the plan should be updated based on state change."""
        self.steps_since_update += 1

        # Force update if too much time has passed
        if self.steps_since_update >= self.max_plan_frequency:
            return True

        # Don't update too frequently
        if self.steps_since_update < self.min_plan_frequency:
            return False

        # Update if we don't have a previous goal vector
        if self.last_goal_vector is None:
            return True

        # Compute state change (simplified heuristic)
        # In practice, you might want more sophisticated state change detection
        if hasattr(self, '_last_state') and self._last_state is not None:
            state_change = torch.norm(current_state - self._last_state).item()
            if state_change > self.adaptation_threshold:
                return True

        self._last_state = current_state.clone()
        return False

    def forward(self, observations: torch.Tensor, state: Optional[Any] = None, return_state: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive planning."""
        result = super().forward(observations, state, return_state)

        # Update tracking
        if self.should_update_plan(observations[-1]):  # Use last observation
            self.last_goal_vector = result['goal_vectors'].clone()
            self.steps_since_update = 0
        elif self.last_goal_vector is not None:
            # Use previous goal vector if not updating
            result['goal_vectors'] = self.last_goal_vector

        return result


def create_planner(
    planner_type: str,
    observation_size: int,
    goal_vector_dim: int = 12,
    **kwargs
) -> StrategicPlanner:
    """
    Factory function to create different types of planners.

    Args:
        planner_type: Type of planner ('basic', 'memory_replay', 'adaptive')
        observation_size: Size of observations
        goal_vector_dim: Dimension of goal vectors
        **kwargs: Additional arguments

    Returns:
        Planner instance
    """
    if planner_type == 'basic':
        return StrategicPlanner(
            observation_size=observation_size,
            goal_vector_dim=goal_vector_dim,
            **kwargs
        )
    elif planner_type == 'memory_replay':
        return PlannerWithMemoryReplay(
            observation_size=observation_size,
            goal_vector_dim=goal_vector_dim,
            **kwargs
        )
    elif planner_type == 'adaptive':
        return AdaptivePlanner(
            observation_size=observation_size,
            goal_vector_dim=goal_vector_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")
