import unittest
import torch
from src.components.controller import Controller

class TestController(unittest.TestCase):

    def setUp(self):
        # Define common dummy hyperparameters for tests
        self.state_dim = 107
        self.goal_dim = 32
        self.hidden_units = [256, 256]
        self.action_dim = 8
        self.batch_size = 4

    def test_controller_initialization(self):
        # Test if the Controller can be initialized without errors
        controller = Controller(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            hidden_units=self.hidden_units,
            action_dim=self.action_dim
        )
        self.assertIsInstance(controller, Controller)
        # Check if the sequential network is created
        self.assertIsInstance(controller.net, torch.nn.Sequential)

    def test_controller_forward_pass_shape(self):
        # Test the output shape of the forward pass
        controller = Controller(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            hidden_units=self.hidden_units,
            action_dim=self.action_dim
        )
        dummy_state = torch.randn(self.batch_size, self.state_dim)
        dummy_goal = torch.randn(self.batch_size, self.goal_dim)

        output_action = controller(dummy_state, dummy_goal)
        expected_shape = (self.batch_size, self.action_dim)
        self.assertEqual(output_action.shape, expected_shape,
                         f"Output shape mismatch! Expected {expected_shape}, got {output_action.shape}")

    def test_controller_output_range(self):
        # Test if the output actions are within the expected [-1, 1] range due to Tanh
        controller = Controller(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            hidden_units=self.hidden_units,
            action_dim=self.action_dim
        )
        dummy_state = torch.randn(self.batch_size, self.state_dim)
        dummy_goal = torch.randn(self.batch_size, self.goal_dim)

        output_action = controller(dummy_state, dummy_goal)
        self.assertTrue(torch.all(output_action >= -1.0) and torch.all(output_action <= 1.0),
                        "Output values are not within the [-1, 1] range.")

    def test_controller_device_handling(self):
        # Determine the best available device for testing
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        controller = Controller(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            hidden_units=self.hidden_units,
            action_dim=self.action_dim
        )
        controller.to(device)

        # Create dummy inputs on the determined device
        dummy_state = torch.randn(self.batch_size, self.state_dim, device=device)
        dummy_goal = torch.randn(self.batch_size, self.goal_dim, device=device)

        # Perform forward pass
        output_action = controller(dummy_state, dummy_goal)

        # Assert that the output is on the correct device
        self.assertEqual(output_action.device.type, device.type, f"Output not on expected device type {device.type}")

        # Test with goal on a different device (should be moved internally)
        if device.type == "cpu": # If main device is CPU, try moving goal to a non-CPU device if available
            if torch.cuda.is_available():
                alt_device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                alt_device = torch.device("mps")
            else:
                alt_device = None # No alternative device to test
        else: # If main device is GPU (cuda/mps), try moving goal to CPU
            alt_device = torch.device("cpu")

        if alt_device:
            dummy_state_main_device = torch.randn(self.batch_size, self.state_dim, device=device)
            dummy_goal_alt_device = torch.randn(self.batch_size, self.goal_dim, device=alt_device)

            output_action_alt_goal = controller(dummy_state_main_device, dummy_goal_alt_device)
            self.assertEqual(output_action_alt_goal.device.type, device.type, f"Output not on expected device type {device.type} when goal was on {alt_device.type}")

if __name__ == '__main__':
    unittest.main()