
import torch.nn as nn

class StateEncoder(nn.Module):
    """
    A simple MLP to encode the environment state.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        """
        Initializes the StateEncoder.

        Args:
            input_dim (int): The dimensionality of the input state.
            output_dim (int): The dimensionality of the encoded state.
            hidden_dim (int): The size of the hidden layers.
        """
        super(StateEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The encoded state tensor.
        """
        return self.network(x)
