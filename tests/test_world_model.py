import unittest
import torch
from src.components.world_model import WorldModel
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class TestWorldModel(unittest.TestCase):

    def setUp(self):
        self.observation_shape = (3, 64, 64) # Example: RGB image
        self.action_dim = 4 # Example: 4 continuous actions
        self.latent_dim = 32
        self.reward_dim = 1
        self.batch_size = 2

        # Define a dummy xlstm_config dictionary
        self.xlstm_config = {
            "mlstm_block": {
                "mlstm": {
                    "conv1d_kernel_size": 4,
                    "qkv_proj_blocksize": 4,
                    "num_heads": 4
                }
            },
            "slstm_block": {
                "slstm": {
                "backend": "cuda" if torch.cuda.is_available() else "vanilla", # Use "vanilla" for CPU
                "num_heads": 4,
                "conv1d_kernel_size": 4,
                "bias_init": "powerlaw_blockdependent",
            },
                "feedforward": {"proj_factor": 1.3, "act_fn": "gelu"},
            },
            "context_length": 256,
            "num_blocks": 2,
            "slstm_at": [1],
        }

        # Determine the best available device for testing
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def test_world_model_initialization(self):
        world_model = WorldModel(
            observation_shape=self.observation_shape,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            reward_dim=self.reward_dim,
            xlstm_config=self.xlstm_config
        ).to(self.device)
        self.assertIsInstance(world_model, WorldModel)
        self.assertIsInstance(world_model.encoder, torch.nn.Sequential)
        self.assertIsInstance(world_model.dynamics_model, xLSTMBlockStack)
        self.assertIsInstance(world_model.reward_predictor, torch.nn.Sequential)
        self.assertIsInstance(world_model.observation_predictor, torch.nn.Sequential)

    def test_world_model_forward_pass_shapes(self):
        world_model = WorldModel(
            observation_shape=self.observation_shape,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            reward_dim=self.reward_dim,
            xlstm_config=self.xlstm_config
        ).to(self.device)

        dummy_observation = torch.randn(self.batch_size, *self.observation_shape, device=self.device)
        dummy_action = torch.randn(self.batch_size, self.action_dim, device=self.device)
        dummy_previous_latent_state = torch.randn(self.batch_size, self.latent_dim, device=self.device)

        predicted_latent_state, predicted_reward, reconstructed_observation, encoded_latent_state, new_hidden_state = \
            world_model(dummy_observation, dummy_action, dummy_previous_latent_state)

        self.assertEqual(predicted_latent_state.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(predicted_reward.shape, (self.batch_size, self.reward_dim))
        self.assertEqual(reconstructed_observation.shape, (self.batch_size, *self.observation_shape))
        self.assertEqual(encoded_latent_state.shape, (self.batch_size, self.latent_dim))
        self.assertIsInstance(new_hidden_state, dict)
        self.assertTrue(len(new_hidden_state) > 0)

    def test_world_model_device_handling(self):
        world_model = WorldModel(
            observation_shape=self.observation_shape,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            reward_dim=self.reward_dim,
            xlstm_config=self.xlstm_config
        )
        world_model.to(self.device)

        dummy_observation = torch.randn(self.batch_size, *self.observation_shape, device=self.device)
        dummy_action = torch.randn(self.batch_size, self.action_dim, device=self.device)
        dummy_previous_latent_state = torch.randn(self.batch_size, self.latent_dim, device=self.device)

        predicted_latent_state, predicted_reward, reconstructed_observation, encoded_latent_state, new_hidden_state = \
            world_model(dummy_observation, dummy_action, dummy_previous_latent_state)

        self.assertEqual(predicted_latent_state.device.type, self.device.type)
        self.assertEqual(predicted_reward.device.type, self.device.type)
        self.assertEqual(reconstructed_observation.device.type, self.device.type)
        self.assertEqual(encoded_latent_state.device.type, self.device.type)
        
        for block_key, block_state in new_hidden_state.items():
            for state_key, state_value in block_state.items():
                if isinstance(state_value, torch.Tensor):
                    self.assertEqual(state_value.device.type, self.device.type)
                elif isinstance(state_value, tuple):
                    for tensor_in_tuple in state_value:
                        self.assertEqual(tensor_in_tuple.device.type, self.device.type)

if __name__ == '__main__':
    unittest.main()