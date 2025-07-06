import torch
import torch.nn as nn
from typing import List

class Controller(nn.Module):
    """
    The Controller (Muscle Memory) component.

    A simple MLP that takes the current environment state and a latent goal from
    the Planner, and outputs a continuous action vector.
    """
    def __init__(self, state_dim: int, goal_dim: int, hidden_units: List[int], action_dim: int):
        """
        Initializes the Controller network.

        Args:
            state_dim (int): The dimension of the environment state vector.
            goal_dim (int): The dimension of the latent goal vector.
            hidden_units (List[int]): A list of integers defining the number of units in each hidden layer.
            action_dim (int): The dimension of the output action vector.
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

        layers = []
        input_dim = state_dim + goal_dim
        
        # Dynamically create the hidden layers
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units
            
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh()) # Scale actions to [-1, 1]

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the Controller.

        Args:
            state (torch.Tensor): The current environment state. Shape: (batch_size, state_dim)
            goal (torch.Tensor): The latent goal vector. Shape: (batch_size, goal_dim)

        Returns:
            torch.Tensor: The output action vector. Shape: (batch_size, action_dim)
        """
        # Ensure tensors are on the same device
        if state.device != goal.device:
            goal = goal.to(state.device)

        # Concatenate state and goal to form the input
        combined_input = torch.cat([state, goal], dim=-1)
        return self.net(combined_input)


