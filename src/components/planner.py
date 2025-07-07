import torch
import torch.nn as nn
from typing import Tuple, Dict
from xlstm import xLSTMBlockStack
from xlstm import (
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class Planner(nn.Module):
    """
    The Planner (Strategist) component.

    Processes sequences of latent states from the World Model using an xLSTMBlockStack
    and outputs parameters for a Mixture Density Network (MDN) to propose
    high-level latent goals.
    """
    def __init__(self,
                 encoder_dim: int,
                 goal_dim: int,
                 mdn_components: int,
                 xlstm_config: Dict # Configuration dictionary for xLSTMBlockStackConfig
                ):
        """
        Initializes the Planner network.

        Args:
            encoder_dim (int): The dimension of the encoded state vector from the StateEncoder.
            goal_dim (int): The dimension of the latent goal vector.
            mdn_components (int): The number of mixture components (k) for the MDN.
            xlstm_config (Dict): Configuration dictionary to build xLSTMBlockStackConfig.
                                 This dict should contain parameters like num_blocks, etc.
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.goal_dim = goal_dim
        self.mdn_components = mdn_components

        # Construct xLSTMBlockStackConfig from the provided dictionary
        mlstm_block_cfg = None
        if 'mlstm_block' in xlstm_config:
            mlstm_block_cfg = mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(**xlstm_config['mlstm_block']['mlstm'])
            )

        slstm_block_cfg = None
        if 'slstm_block' in xlstm_config:
            slstm_block_cfg = sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(**xlstm_config['slstm_block']['slstm']),
                feedforward=FeedForwardConfig(**xlstm_config['slstm_block']['feedforward'])
            )

        xlstm_block_stack_cfg = xLSTMBlockStackConfig(
            mlstm_block=mlstm_block_cfg,
            slstm_block=slstm_block_cfg,
            context_length=xlstm_config.get('context_length', 256),
            num_blocks=xlstm_config.get('num_blocks', 1),
            embedding_dim=encoder_dim,
            slstm_at=xlstm_config.get('slstm_at', []),
        )

        # xLSTM Body
        self.xlstm_stack = xLSTMBlockStack(xlstm_block_stack_cfg)

        # MDN Head
        self.pi_head = nn.Linear(encoder_dim, mdn_components)
        self.mu_head = nn.Linear(encoder_dim, mdn_components * goal_dim)
        self.sigma_head = nn.Linear(encoder_dim, mdn_components * goal_dim)

    def forward(self, 
                encoded_state_sequence: torch.Tensor, 
                hidden_state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        """
        Performs the forward pass through the Planner.

        Args:
            encoded_state_sequence (torch.Tensor): A sequence of encoded states.
                                                  Shape: (batch_size, sequence_length, encoder_dim)
            hidden_state (dict, optional): Initial hidden state for the xLSTM.
                                           Defaults to None (empty dictionary).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
                - pi (torch.Tensor): Mixture weights. Shape: (batch_size, mdn_components)
                - mu (torch.Tensor): Means of the Gaussian components. Shape: (batch_size, mdn_components, goal_dim)
                - sigma (torch.Tensor): Standard deviations of the Gaussian components. Shape: (batch_size, mdn_components, goal_dim)
                - new_hidden_state (dict): The updated hidden state of the xLSTM.
        """
        batch_size, sequence_length, _ = encoded_state_sequence.shape

        if hidden_state is None:
            hidden_state = {}

        xlstm_output = None
        for t in range(sequence_length):
            current_input = encoded_state_sequence[:, t, :].unsqueeze(1) # Process one step at a time
            xlstm_output, hidden_state = self.xlstm_stack.step(current_input, state=hidden_state)

        # Take the output from the last time step for the MDN head
        strategic_context = xlstm_output.squeeze(1) # Remove sequence_length dimension (which is 1)

        # MDN Head calculations
        pi_logits = self.pi_head(strategic_context)
        mu_raw = self.mu_head(strategic_context)
        sigma_raw = self.sigma_head(strategic_context)

        # Apply activations
        pi = torch.softmax(pi_logits, dim=-1)
        mu = mu_raw.view(batch_size, self.mdn_components, self.goal_dim)
        sigma = torch.nn.functional.softplus(sigma_raw).view(batch_size, self.mdn_components, self.goal_dim)

        return pi, mu, sigma, hidden_state
