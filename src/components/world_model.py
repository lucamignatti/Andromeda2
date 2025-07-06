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

class WorldModel(nn.Module):
    """
    The World Model (Dream Engine) component.

    Learns the temporal dynamics of the environment, enabling prediction and imagination.
    """
    def __init__(self,
                 observation_shape: Tuple[int, ...],
                 action_dim: int,
                 latent_dim: int,
                 reward_dim: int,
                 xlstm_config: Dict # Configuration dictionary for xLSTMBlockStackConfig
                ):
        """
        Initializes the World Model network.

        Args:
            observation_shape (Tuple[int, ...]): The shape of the environment observation (e.g., (3, 64, 64) for an image).
            action_dim (int): The dimension of the action space.
            latent_dim (int): The dimension of the latent state vector.
            reward_dim (int): The dimension of the reward (usually 1).
            xlstm_config (Dict): Configuration dictionary to build xLSTMBlockStackConfig.
        """
        super().__init__()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.reward_dim = reward_dim

        # Encoder (MLP) - Compresses high-dimensional observation into latent state
        # Assuming a flattened observation for simplicity for now. Will need to adjust for images.
        # For now, let's assume observation_shape is (C, H, W) and we flatten it.
        flattened_observation_dim = 1
        for dim in observation_shape:
            flattened_observation_dim *= dim

        self.encoder = nn.Sequential(
            nn.Linear(flattened_observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2) # *2 for mean and logvar of probabilistic latent state
        )

        # Dynamics Model (xLSTM) - Predicts next latent state
        # Input to xLSTM will be concatenated (previous_latent_state + action)
        dynamics_input_dim = latent_dim + action_dim

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
            embedding_dim=dynamics_input_dim, # xLSTM embedding dim is the input dim
            slstm_at=xlstm_config.get('slstm_at', []),
        )
        self.dynamics_model = xLSTMBlockStack(xlstm_block_stack_cfg)

        # Projection layer to map xLSTM output back to latent_dim
        self.dynamics_output_projection = nn.Linear(dynamics_input_dim, latent_dim * 2) # *2 for mean and logvar

        # Reward Predictor (MLP) - Forecasts future rewards from latent state
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, reward_dim)
        )

        # Observation Predictor (Decoder) (MLP) - Reconstructs observation from latent state
        self.observation_predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, flattened_observation_dim)
        )

    def forward(self,
                observation: torch.Tensor,
                action: torch.Tensor,
                previous_latent_state: torch.Tensor,
                hidden_state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Performs a single step prediction in the World Model.

        Args:
            observation (torch.Tensor): Current environment observation. Shape: (batch_size, *observation_shape)
            action (torch.Tensor): Action taken at the previous step. Shape: (batch_size, action_dim)
            previous_latent_state (torch.Tensor): Latent state from the previous step. Shape: (batch_size, latent_dim)
            hidden_state (dict, optional): Hidden state for the xLSTM dynamics model. Defaults to None (empty dictionary).

        Returns:
            Tuple containing:
                - predicted_latent_mean (torch.Tensor)
                - predicted_latent_logvar (torch.Tensor)
                - predicted_reward (torch.Tensor)
                - reconstructed_observation (torch.Tensor)
                - encoded_latent_mean (torch.Tensor)
                - encoded_latent_logvar (torch.Tensor)
                - encoded_latent_state (torch.Tensor): Sampled latent state from encoder.
                - new_hidden_state (dict)
        """
        batch_size = observation.shape[0]

        # 1. Encode the current observation
        flat_observation = observation.view(batch_size, -1)
        encoded_latent_params = self.encoder(flat_observation)
        encoded_latent_mean, encoded_latent_logvar = torch.chunk(encoded_latent_params, 2, dim=-1)
        
        # Sample from the encoded distribution (Reparameterization Trick)
        std = torch.exp(0.5 * encoded_latent_logvar)
        eps = torch.randn_like(std)
        encoded_latent_state = encoded_latent_mean + eps * std

        # 2. Dynamics Model (xLSTM) - Predict next latent state
        dynamics_input = torch.cat([previous_latent_state, action], dim=-1).unsqueeze(1)

        if hidden_state is None:
            hidden_state = {}

        predicted_latent_output, new_hidden_state = self.dynamics_model.step(dynamics_input, state=hidden_state)
        
        projected_dynamics_output = self.dynamics_output_projection(predicted_latent_output.squeeze(1))
        predicted_latent_mean, predicted_latent_logvar = torch.chunk(projected_dynamics_output, 2, dim=-1)

        # For prediction, we use the mean of the dynamics model's output
        predicted_latent_state_for_predictors = predicted_latent_mean

        # 3. Reward Predictor
        predicted_reward = self.reward_predictor(predicted_latent_state_for_predictors)

        # 4. Observation Predictor (Decoder)
        reconstructed_observation = self.observation_predictor(predicted_latent_state_for_predictors)
        reconstructed_observation = reconstructed_observation.view(batch_size, *self.observation_shape)

        return (
            predicted_latent_mean,
            predicted_latent_logvar,
            predicted_reward,
            reconstructed_observation,
            encoded_latent_mean,
            encoded_latent_logvar,
            encoded_latent_state,
            new_hidden_state
        )
