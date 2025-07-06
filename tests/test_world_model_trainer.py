import unittest
import torch
import yaml
import os
import shutil
import tempfile
from src.training.world_model_trainer import WorldModelTrainer
from src.training.base_trainer import BaseTrainer
from src.utils.replay_buffer import ReplayBuffer
from tests.utils.dummy_env import DummyEnv
from src.components.world_model import WorldModel
import torch.optim as optim

class TestWorldModelTrainer(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and dummy config files for each test."""
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
                'context_length': 16,
                'num_blocks': 1,
                'mlstm_block': {
                    'mlstm': {'num_heads': 4}
                }
            }
        }
        self.train_config = {
            'learning_rate': 1e-4,
            'replay_capacity': 100,
            'initial_data_trajectories': 5,
            'epochs': 2,
            'batch_size': 2,
            'grad_clip': 100.0,
            'checkpoint_interval': 1,
            'kl_beta': 1.0,
            'max_trajectory_length': 20,
        }

        with open(os.path.join(self.configs_dir, 'global.yaml'), 'w') as f:
            yaml.dump(self.global_config, f)
        with open(os.path.join(self.configs_dir, 'world_model.yaml'), 'w') as f:
            yaml.dump(self.wm_config, f)
        with open(os.path.join(self.training_configs_dir, 'stage1_world_model.yaml'), 'w') as f:
            yaml.dump(self.train_config, f)

        # --- Monkeypatch the __init__ method to use our temp files ---
        self.original_init = WorldModelTrainer.__init__
        def mock_init(trainer_self):
            trainer_self.global_config = self.global_config
            BaseTrainer.__init__(trainer_self, global_config_path=os.path.join(self.configs_dir, 'global.yaml'))
            
            trainer_self.wm_config = self.wm_config
            trainer_self.train_config = self.train_config
            
            trainer_self.world_model = WorldModel(
                observation_shape=tuple(trainer_self.global_config['observation_shape']),
                action_dim=trainer_self.global_config['action_dim'],
                latent_dim=trainer_self.wm_config['latent_dim'],
                reward_dim=trainer_self.global_config['reward_dim'],
                xlstm_config=trainer_self.wm_config['xlstm_config']
            ).to(trainer_self.device)

            trainer_self.optimizer = optim.Adam(trainer_self.world_model.parameters(), lr=trainer_self.train_config['learning_rate'])
            trainer_self.checkpoint_dir = trainer_self._create_checkpoint_dir('stage1_world_model')
            
            trainer_self.replay_buffer = ReplayBuffer(
                capacity=trainer_self.train_config['replay_capacity'],
                sequence_length=trainer_self.wm_config['xlstm_config']['context_length']
            )
            
            trainer_self.env = DummyEnv(
                observation_shape=tuple(trainer_self.global_config['observation_shape']),
                action_dim=trainer_self.global_config['action_dim'],
                reward_dim=trainer_self.global_config['reward_dim']
            )

        WorldModelTrainer.__init__ = mock_init
        self.trainer = WorldModelTrainer()

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)
        WorldModelTrainer.__init__ = self.original_init

    def test_01_initialization(self):
        """Test if the trainer and its components are initialized correctly."""
        self.assertIsInstance(self.trainer, WorldModelTrainer)
        self.assertIsNotNone(self.trainer.world_model)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.replay_buffer)
        self.assertEqual(self.trainer.replay_buffer.capacity, self.train_config['replay_capacity'])
        self.assertTrue(os.path.exists(self.trainer.checkpoint_dir))

    def test_02_collect_initial_data(self):
        """Test if the replay buffer is populated correctly."""
        self.trainer.collect_initial_data()
        self.assertEqual(len(self.trainer.replay_buffer), self.train_config['initial_data_trajectories'])
        
        obs, act, rew, next_obs = self.trainer.replay_buffer.sample(1)
        self.assertEqual(obs.shape, (1, self.wm_config['xlstm_config']['context_length'], *self.global_config['observation_shape']))
        self.assertEqual(act.shape, (1, self.wm_config['xlstm_config']['context_length'], self.global_config['action_dim']))

    def test_03_train_step(self):
        """Test that a single training step updates the model weights."""
        self.trainer.collect_initial_data()
        
        initial_weights = self.trainer.world_model.encoder[0].weight.clone().detach()

        self.trainer.train_config['epochs'] = 1
        self.trainer.train()

        updated_weights = self.trainer.world_model.encoder[0].weight.clone().detach()

        self.assertFalse(torch.equal(initial_weights, updated_weights))
        self.assertTrue(torch.all(torch.isfinite(updated_weights)))

    def test_04_save_checkpoint(self):
        """Test if a checkpoint is saved correctly."""
        self.trainer.collect_initial_data()
        self.trainer.train_config['epochs'] = 1
        self.trainer.train_config['checkpoint_interval'] = 1
        
        self.trainer.train()

        epoch_dir = os.path.join(self.trainer.checkpoint_dir)
        self.assertTrue(os.path.exists(epoch_dir))
        
        saved_files = os.listdir(epoch_dir)
        self.assertEqual(len(saved_files), 1)
        
        checkpoint_path = os.path.join(epoch_dir, saved_files[0])
        self.assertTrue(os.path.exists(checkpoint_path))

        checkpoint = torch.load(checkpoint_path)
        self.assertIn('epoch', checkpoint)
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
        self.assertEqual(checkpoint['epoch'], 1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
