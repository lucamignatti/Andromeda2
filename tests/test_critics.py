import unittest
import torch
from src.components.critics import ValueIntrinsicCritic, TemporalC51ExtrinsicCritic

class TestCritics(unittest.TestCase):

    def setUp(self):
        self.encoder_dim = 256 # Changed from latent_dim
        self.goal_dim = 256    # Updated to match the new architecture
        self.hidden_units = [256, 256]
        self.batch_size = 4

        # C51 specific params
        self.num_atoms = 51
        self.v_min = -10.0
        self.v_max = 10.0
        self.temporal_horizons = [16, 64, 256]

        # Determine the best available device for testing
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def test_value_intrinsic_critic_initialization(self):
        critic = ValueIntrinsicCritic(
            encoder_dim=self.encoder_dim, # Changed from latent_dim
            goal_dim=self.goal_dim,
            hidden_units=self.hidden_units
        ).to(self.device)
        self.assertIsInstance(critic, ValueIntrinsicCritic)
        self.assertIsInstance(critic.net, torch.nn.Sequential)

    def test_value_intrinsic_critic_forward_pass_shape(self):
        critic = ValueIntrinsicCritic(
            encoder_dim=self.encoder_dim, # Changed from latent_dim
            goal_dim=self.goal_dim,
            hidden_units=self.hidden_units
        ).to(self.device)
        
        dummy_encoded_state = torch.randn(self.batch_size, self.encoder_dim, device=self.device) # Changed variable name
        dummy_goal = torch.randn(self.batch_size, self.goal_dim, device=self.device)

        output_value = critic(dummy_encoded_state, dummy_goal)
        expected_shape = (self.batch_size, 1)
        self.assertEqual(output_value.shape, expected_shape,
                         f"Output shape mismatch! Expected {expected_shape}, got {output_value.shape}")

    def test_value_intrinsic_critic_device_handling(self):
        critic = ValueIntrinsicCritic(
            encoder_dim=self.encoder_dim, # Changed from latent_dim
            goal_dim=self.goal_dim,
            hidden_units=self.hidden_units
        )
        critic.to(self.device)

        dummy_encoded_state = torch.randn(self.batch_size, self.encoder_dim, device=self.device) # Changed variable name
        dummy_goal = torch.randn(self.batch_size, self.goal_dim, device=self.device)

        output_value = critic(dummy_encoded_state, dummy_goal)
        self.assertEqual(output_value.device.type, self.device.type, f"Output not on expected device type {self.device.type}")

        # Test with goal on a different device (should be moved internally)
        if self.device.type == "cpu":
            if torch.cuda.is_available():
                alt_device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                alt_device = torch.device("mps")
            else:
                alt_device = None # No alternative device to test
        else:
            alt_device = torch.device("cpu")

        if alt_device:
            dummy_encoded_state_main_device = torch.randn(self.batch_size, self.encoder_dim, device=self.device) # Changed variable name
            dummy_goal_alt_device = torch.randn(self.batch_size, self.goal_dim, device=alt_device)

            output_value_alt_goal = critic(dummy_encoded_state_main_device, dummy_goal_alt_device)
            self.assertEqual(output_value_alt_goal.device.type, self.device.type, f"Output not on expected device type {self.device.type} when goal was on {alt_device.type}")

    def test_temporal_c51_extrinsic_critic_initialization(self):
        critic = TemporalC51ExtrinsicCritic(
            encoder_dim=self.encoder_dim, # Changed from latent_dim
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            temporal_horizons=self.temporal_horizons,
            hidden_units=self.hidden_units
        ).to(self.device)
        self.assertIsInstance(critic, TemporalC51ExtrinsicCritic)
        self.assertIsInstance(critic.net, torch.nn.Sequential)
        self.assertTrue(hasattr(critic, 'support'))
        self.assertEqual(critic.support.shape, (self.num_atoms,))

    def test_temporal_c51_extrinsic_critic_forward_pass_shape(self):
        critic = TemporalC51ExtrinsicCritic(
            encoder_dim=self.encoder_dim, # Changed from latent_dim
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            temporal_horizons=self.temporal_horizons,
            hidden_units=self.hidden_units
        ).to(self.device)

        dummy_encoded_state = torch.randn(self.batch_size, self.encoder_dim, device=self.device) # Changed variable name

        output_logits = critic(dummy_encoded_state)
        expected_shape = (self.batch_size, len(self.temporal_horizons), self.num_atoms)
        self.assertEqual(output_logits.shape, expected_shape,
                         f"Output shape mismatch! Expected {expected_shape}, got {output_logits.shape}")

    def test_temporal_c51_extrinsic_critic_device_handling(self):
        critic = TemporalC51ExtrinsicCritic(
            encoder_dim=self.encoder_dim, # Changed from latent_dim
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            temporal_horizons=self.temporal_horizons,
            hidden_units=self.hidden_units
        )
        critic.to(self.device)

        dummy_encoded_state = torch.randn(self.batch_size, self.encoder_dim, device=self.device) # Changed variable name

        output_logits = critic(dummy_encoded_state)
        self.assertEqual(output_logits.device.type, self.device.type, f"Output not on expected device type {self.device.type}")

        # Test support buffer device
        self.assertEqual(critic.support.device.type, self.device.type, f"Support buffer not on expected device type {self.device.type}")

if __name__ == '__main__':
    unittest.main()
