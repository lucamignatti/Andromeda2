import unittest
import torch
import yaml
import os
import shutil
import tempfile
from src.training.controller_trainer import ControllerTrainer
from src.components.world_model import WorldModel
from src.components.controller import Controller
from src.components.critics import ValueIntrinsicCritic
from src.training.base_trainer import BaseTrainer
from tests.utils.dummy_env import DummyEnv
import torch.optim as optim

class TestControllerTrainer(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory, dummy config files, and a dummy world model checkpoint."""
        self.temp_dir = tempfile.mkdtemp()
        self.configs_dir = os.path.join(self.temp_dir, 'configs')
        self.training_configs_dir = os.path.join(self.configs_dir, 'training')
        self.checkpoints_dir = os.path.join(self.temp_dir, 'checkpoints')
        os.makedirs(self.training_configs_dir)
        os.makedirs(self.checkpoints_dir)

        # --- Create Dummy Config Files ---
        self.global_config = {
            'device': 'cpu',
            'checkpoint_dir': self.checkpoints_dir,
            'observation_shape': [3, 64, 64],
            'action_dim': 4,
            'reward_dim': 1,
        }
        self.wm_config = {
            'latent_dim': 256,
            'xlstm_config': {
                'context_length': 16, 'num_blocks': 1,
                'mlstm_block': {'mlstm': {'num_heads': 4}}
            }
        }
        self.ctrl_config = {'hidden_units': [128, 128]}
        self.critic_config = {'hidden_units': [128, 128]}
        self.train_config = {
            'learning_rate_controller': 1e-4,
            'learning_rate_critic': 1e-4,
            'epochs': 2,
            'gamma': 0.99,
            'goal_horizon': 5,
            'train_horizon': 10,
            'checkpoint_interval': 1,
        }

        # --- Create Dummy World Model Checkpoint ---
        self.wm_checkpoint_path = os.path.join(self.checkpoints_dir, 'dummy_wm.pt')
        dummy_wm = WorldModel(
            observation_shape=tuple(self.global_config['observation_shape']),
            action_dim=self.global_config['action_dim'],
            latent_dim=self.wm_config['latent_dim'],
            reward_dim=self.global_config['reward_dim'],
            xlstm_config=self.wm_config['xlstm_config']
        )
        torch.save({'model_state_dict': dummy_wm.state_dict()}, self.wm_checkpoint_path)

        # --- Monkeypatch the __init__ method ---
        self.original_init = ControllerTrainer.__init__
        def mock_init(trainer_self, world_model_checkpoint_path):
            trainer_self.global_config = self.global_config
            BaseTrainer.__init__(trainer_self, global_config_path=None)

            trainer_self.ctrl_config = self.ctrl_config
            trainer_self.train_config = self.train_config
            trainer_self.wm_config = self.wm_config
            trainer_self.critic_config = self.critic_config

            # Manually set device and checkpoint_dir because we are not loading the global config
            trainer_self.device = torch.device(self.global_config['device'])
            trainer_self.checkpoint_dir = self.global_config['checkpoint_dir']

            trainer_self.world_model = trainer_self._load_world_model(world_model_checkpoint_path)

            trainer_self.controller = Controller(
                state_dim=trainer_self.wm_config['latent_dim'],
                goal_dim=trainer_self.wm_config['latent_dim'],
                hidden_units=trainer_self.ctrl_config['hidden_units'],
                action_dim=trainer_self.global_config['action_dim']
            ).to(trainer_self.device)

            trainer_self.intrinsic_critic = ValueIntrinsicCritic(
                latent_dim=trainer_self.wm_config['latent_dim'],
                goal_dim=trainer_self.wm_config['latent_dim'],
                hidden_units=trainer_self.critic_config['hidden_units']
            ).to(trainer_self.device)

            trainer_self.controller_optimizer = optim.Adam(trainer_self.controller.parameters(), lr=trainer_self.train_config['learning_rate_controller'])
            trainer_self.critic_optimizer = optim.Adam(trainer_self.intrinsic_critic.parameters(), lr=trainer_self.train_config['learning_rate_critic'])

            trainer_self.checkpoint_dir = trainer_self._create_checkpoint_dir('stage2_controller')
            trainer_self.env = DummyEnv(
                observation_shape=tuple(self.global_config['observation_shape']),
                action_dim=self.global_config['action_dim'],
                reward_dim=self.global_config['reward_dim']
            )

        ControllerTrainer.__init__ = mock_init
        self.trainer = ControllerTrainer(self.wm_checkpoint_path)


    def tearDown(self):
        """Clean up the temporary directory and restore original methods."""
        shutil.rmtree(self.temp_dir)
        ControllerTrainer.__init__ = self.original_init

    def test_01_initialization(self):
        """Test if the trainer and its components are initialized correctly."""
        self.assertIsInstance(self.trainer, ControllerTrainer)
        self.assertIsNotNone(self.trainer.world_model)
        self.assertIsNotNone(self.trainer.controller)
        self.assertIsNotNone(self.trainer.intrinsic_critic)
        self.assertIsNotNone(self.trainer.controller_optimizer)
        self.assertIsNotNone(self.trainer.critic_optimizer)
        self.assertTrue(os.path.exists(self.trainer.checkpoint_dir))

    def test_02_train_step(self):
        """Test that a single training step updates both controller and critic weights."""
        # Get initial weights
        initial_ctrl_weights = self.trainer.controller.net[0].weight.clone().detach()
        initial_critic_weights = self.trainer.intrinsic_critic.net[0].weight.clone().detach()

        # Run one training epoch
        self.trainer.train_config['epochs'] = 1
        self.trainer.train()

        # Get updated weights
        updated_ctrl_weights = self.trainer.controller.net[0].weight.clone().detach()
        updated_critic_weights = self.trainer.intrinsic_critic.net[0].weight.clone().detach()

        # Check that weights have changed
        self.assertFalse(torch.equal(initial_ctrl_weights, updated_ctrl_weights))
        self.assertFalse(torch.equal(initial_critic_weights, updated_critic_weights))
        self.assertTrue(torch.all(torch.isfinite(updated_ctrl_weights)))
        self.assertTrue(torch.all(torch.isfinite(updated_critic_weights)))

    def test_03_save_checkpoint(self):
        """Test if a checkpoint is saved correctly."""
        self.trainer.train_config['epochs'] = 1
        self.trainer.train_config['checkpoint_interval'] = 1
        self.trainer.train()

        saved_files = os.listdir(self.trainer.checkpoint_dir)
        self.assertEqual(len(saved_files), 1)

        checkpoint_path = os.path.join(self.trainer.checkpoint_dir, saved_files[0])
        self.assertTrue(os.path.exists(checkpoint_path))

        # Load the checkpoint and verify its contents
        checkpoint = torch.load(checkpoint_path)
        self.assertIn('epoch', checkpoint)
        self.assertIn('controller_state_dict', checkpoint)
        self.assertIn('critic_state_dict', checkpoint)
        self.assertIn('controller_optimizer_state_dict', checkpoint)
        self.assertIn('critic_optimizer_state_dict', checkpoint)
        self.assertEqual(checkpoint['epoch'], 1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
