import unittest
import torch
from src.components.planner import Planner
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class TestPlanner(unittest.TestCase):

    def setUp(self):
        self.latent_dim = 256
        self.goal_dim = 32
        # xlstm_hidden_size is now embedding_dim in xLSTMBlockStackConfig
        self.xlstm_embedding_dim = self.latent_dim # xLSTM output dim should match latent_dim
        self.xlstm_num_blocks = 2 # Corresponds to num_layers in previous setup
        self.mdn_components = 5
        self.batch_size = 4
        self.sequence_length = 1

        # Define a dummy xlstm_config dictionary that mirrors the structure
        # required by xLSTMBlockStackConfig and its nested configs.
        # This should be consistent with how we'll define it in configs/planner.yaml
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
            "num_blocks": self.xlstm_num_blocks,
            "slstm_at": [1], # Example: sLSTM at block 1
            # embedding_dim is set to latent_dim inside Planner's __init__
        }

        # Determine the best available device for testing
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def test_planner_initialization(self):
        planner = Planner(
            latent_dim=self.latent_dim,
            goal_dim=self.goal_dim,
            mdn_components=self.mdn_components,
            xlstm_config=self.xlstm_config
        ).to(self.device)
        self.assertIsInstance(planner, Planner)
        self.assertIsInstance(planner.xlstm_stack, xLSTMBlockStack)
        self.assertIsInstance(planner.pi_head, torch.nn.Linear)
        self.assertIsInstance(planner.mu_head, torch.nn.Linear)
        self.assertIsInstance(planner.sigma_head, torch.nn.Linear)

    def test_planner_forward_pass_shapes(self):
        planner = Planner(
            latent_dim=self.latent_dim,
            goal_dim=self.goal_dim,
            mdn_components=self.mdn_components,
            xlstm_config=self.xlstm_config
        ).to(self.device)

        dummy_latent_sequence = torch.randn(self.batch_size, self.sequence_length, self.latent_dim, device=self.device)

        pi, mu, sigma, hidden_state = planner(dummy_latent_sequence)

        self.assertEqual(pi.shape, (self.batch_size, self.mdn_components))
        self.assertEqual(mu.shape, (self.batch_size, self.mdn_components, self.goal_dim))
        self.assertEqual(sigma.shape, (self.batch_size, self.mdn_components, self.goal_dim))
        self.assertIsInstance(hidden_state, dict)
        self.assertTrue(len(hidden_state) > 0)

    def test_planner_output_properties(self):
        planner = Planner(
            latent_dim=self.latent_dim,
            goal_dim=self.goal_dim,
            mdn_components=self.mdn_components,
            xlstm_config=self.xlstm_config
        ).to(self.device)

        dummy_latent_sequence = torch.randn(self.batch_size, self.sequence_length, self.latent_dim, device=self.device)

        pi, mu, sigma, _ = planner(dummy_latent_sequence)

        # Check pi sums to 1 along the last dimension
        self.assertTrue(torch.allclose(pi.sum(dim=-1), torch.ones(self.batch_size, device=self.device)))
        # Check sigma is positive
        self.assertTrue(torch.all(sigma >= 0))

    def test_planner_device_handling(self):
        planner = Planner(
            latent_dim=self.latent_dim,
            goal_dim=self.goal_dim,
            mdn_components=self.mdn_components,
            xlstm_config=self.xlstm_config
        )
        planner.to(self.device)

        dummy_latent_sequence = torch.randn(self.batch_size, self.sequence_length, self.latent_dim, device=self.device)

        pi, mu, sigma, hidden_state = planner(dummy_latent_sequence)

        self.assertEqual(pi.device.type, self.device.type)
        self.assertEqual(mu.device.type, self.device.type)
        self.assertEqual(sigma.device.type, self.device.type)
        self.assertIsInstance(hidden_state, dict)
        # Check that the tensors within the hidden_state dictionary are on the correct device
        for block_key, block_state in hidden_state.items():
            for state_key, state_value in block_state.items():
                if isinstance(state_value, torch.Tensor):
                    self.assertEqual(state_value.device.type, self.device.type)
                elif isinstance(state_value, tuple):
                    for tensor_in_tuple in state_value:
                        self.assertEqual(tensor_in_tuple.device.type, self.device.type)

if __name__ == '__main__':
    unittest.main()