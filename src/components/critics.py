import torch
import torch.nn as nn
from typing import List

class ValueIntrinsicCritic(nn.Module):
    """
    The Intrinsic Critic (Value-Based) component.

    A simple MLP that predicts the immediate success of the Controller
    in achieving the Planner's current goal.
    """
    def __init__(self, latent_dim: int, goal_dim: int, hidden_units: List[int]):
        """
        Initializes the ValueIntrinsicCritic network.

        Args:
            latent_dim (int): The dimension of the latent state vector from the World Model.
            goal_dim (int): The dimension of the latent goal vector.
            hidden_units (List[int]): A list of integers defining the number of units in each hidden layer.
        """
        super().__init__()

        layers = []
        input_dim = latent_dim + goal_dim

        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units

        # Output layer: single scalar value
        layers.append(nn.Linear(input_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, latent_state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the Intrinsic Critic.

        Args:
            latent_state (torch.Tensor): The current latent state. Shape: (batch_size, latent_dim)
            goal (torch.Tensor): The latent goal vector. Shape: (batch_size, goal_dim)

        Returns:
            torch.Tensor: The predicted intrinsic value. Shape: (batch_size, 1)
        """
        # Ensure tensors are on the same device
        if latent_state.device != goal.device:
            goal = goal.to(latent_state.device)

        combined_input = torch.cat([latent_state, goal], dim=-1)
        return self.net(combined_input)


class TemporalC51ExtrinsicCritic(nn.Module):
    """
    The Extrinsic Critic (Temporal C51) component.

    A sophisticated distributional critic that predicts probability distributions
    over future rewards for multiple time horizons.
    """
    def __init__(self, latent_dim: int, num_atoms: int, v_min: float, v_max: float, temporal_horizons: List[int], hidden_units: List[int]):
        """
        Initializes the TemporalC51ExtrinsicCritic network.

        Args:
            latent_dim (int): The dimension of the latent state vector.
            num_atoms (int): The number of atoms (bins) for the C51 distribution.
            v_min (float): The minimum value for the support of the C51 distribution.
            v_max (float): The maximum value for the support of the C51 distribution.
            temporal_horizons (List[int]): A list of integers representing the different time horizons.
            hidden_units (List[int]): A list of integers defining the number of units in each hidden layer.
        """
        super().__init__()

        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.temporal_horizons = temporal_horizons
        self.num_horizons = len(temporal_horizons)

        layers = []
        input_dim = latent_dim

        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units

        # Output layer: num_horizons * num_atoms logits for the C51 distribution
        layers.append(nn.Linear(input_dim, self.num_horizons * self.num_atoms))

        self.net = nn.Sequential(*layers)

        # Create the support (z_i values) for the C51 distribution
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the Extrinsic Critic.

        Args:
            latent_state (torch.Tensor): The current latent state. Shape: (batch_size, latent_dim)

        Returns:
            torch.Tensor: Logits for the C51 distributions across all horizons.
                          Shape: (batch_size, num_horizons, num_atoms)
        """
        logits = self.net(latent_state)
        
        # Reshape to (batch_size, num_horizons, num_atoms)
        logits = logits.view(-1, self.num_horizons, self.num_atoms)
        
        # Apply softmax to get probabilities for each distribution
        # Note: During training, the loss function will typically handle the softmax
        # or log_softmax for numerical stability. This is just for conceptual clarity.
        # For actual C51, you often work with logits until the final projection.
        # probabilities = torch.softmax(logits, dim=-1)

        return logits
