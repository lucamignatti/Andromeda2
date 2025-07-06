import os
import torch
import torch.optim as optim
import numpy as np
from src.training.base_trainer import BaseTrainer
from src.components.world_model import WorldModel
from src.utils.losses import world_model_loss
from src.utils.replay_buffer import ReplayBuffer
from tests.utils.dummy_env import DummyEnv # Using dummy env for now

class WorldModelTrainer(BaseTrainer):
    """
    Trainer for Stage 1: World Model Training.
    """
    def __init__(self):
        super().__init__()
        self.wm_config = self._load_config('configs/world_model.yaml')
        self.train_config = self._load_config('configs/training/stage1_world_model.yaml')

        self.world_model = WorldModel(
            observation_shape=tuple(self.global_config['observation_shape']),
            action_dim=self.global_config['action_dim'],
            latent_dim=self.wm_config['latent_dim'],
            reward_dim=self.global_config['reward_dim'],
            xlstm_config=self.wm_config['xlstm_config']
        ).to(self.device)

        self.optimizer = optim.Adam(self.world_model.parameters(), lr=self.train_config['learning_rate'])
        self.checkpoint_dir = self._create_checkpoint_dir('stage1_world_model')
        
        self.replay_buffer = ReplayBuffer(
            capacity=self.train_config['replay_capacity'],
            sequence_length=self.wm_config['xlstm_config']['context_length']
        )
        
        self.env = DummyEnv(
            observation_shape=tuple(self.global_config['observation_shape']),
            action_dim=self.global_config['action_dim'],
            reward_dim=self.global_config['reward_dim']
        )

    def collect_initial_data(self):
        print("Collecting initial random data...")
        num_trajectories = self.train_config['initial_data_trajectories']
        for i in range(num_trajectories):
            obs = self.env.reset()
            done = False
            trajectory = []
            while not done:
                action = np.random.randn(self.global_config['action_dim']).astype(np.float32)
                next_obs, reward, done, _ = self.env.step(action)
                trajectory.append((obs, action, reward, next_obs))
                obs = next_obs
                if len(trajectory) >= self.train_config.get('max_trajectory_length', 1000):
                    done = True
            self.replay_buffer.add(trajectory)
        print(f"Collected {len(self.replay_buffer)} trajectories.")

    def train(self):
        self.collect_initial_data()

        for epoch in range(self.train_config['epochs']):
            self.world_model.train()
            
            # Sample a batch from the replay buffer
            obs_batch, act_batch, rew_batch, _ = self.replay_buffer.sample(self.train_config['batch_size'])
            
            # Convert to tensors
            obs_batch = torch.from_numpy(obs_batch).to(self.device)
            act_batch = torch.from_numpy(act_batch).to(self.device)
            rew_batch = torch.from_numpy(rew_batch).to(self.device)

            total_loss = 0
            
            # Initialize hidden state and latent state for each sequence in the batch
            batch_size = obs_batch.shape[0]
            hidden_state = None
            previous_latent_state = torch.zeros(batch_size, self.wm_config['latent_dim']).to(self.device)

            # Process the sequence
            for t in range(self.replay_buffer.sequence_length - 1):
                self.optimizer.zero_grad()

                obs_t = obs_batch[:, t]
                act_t = act_batch[:, t]
                rew_t_plus_1 = rew_batch[:, t + 1]
                obs_t_plus_1 = obs_batch[:, t + 1]

                (
                    pred_latent_mean, pred_latent_logvar, pred_reward, recon_obs,
                    enc_latent_mean, enc_latent_logvar, enc_latent_state, hidden_state
                ) = self.world_model(
                    observation=obs_t_plus_1,
                    action=act_t,
                    previous_latent_state=previous_latent_state,
                    hidden_state=hidden_state
                )

                loss, loss_components = world_model_loss(
                    reconstructed_observation=recon_obs,
                    target_observation=obs_t_plus_1,
                    predicted_reward=pred_reward,
                    target_reward=rew_t_plus_1,
                    predicted_latent_mean=pred_latent_mean,
                    predicted_latent_logvar=pred_latent_logvar,
                    encoded_latent_mean=enc_latent_mean,
                    encoded_latent_logvar=enc_latent_logvar,
                    kl_beta=self.train_config.get('kl_beta', 1.0)
                )

                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.train_config['grad_clip'])
                self.optimizer.step()

                total_loss += loss.item()
                previous_latent_state = enc_latent_state.detach()
                
                # Detach the hidden state to prevent gradient issues
                if hidden_state:
                    for block_type, block_state in hidden_state.items():
                        for layer_name, layer_state in block_state.items():
                            if isinstance(layer_state, tuple):
                                hidden_state[block_type][layer_name] = tuple(s.detach() for s in layer_state)
                            else:
                                hidden_state[block_type][layer_name] = layer_state.detach()

            avg_loss = total_loss / (self.replay_buffer.sequence_length - 1)
            print(f"Epoch {epoch+1}/{self.train_config['epochs']}, Average Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.train_config['checkpoint_interval'] == 0:
                self._save_checkpoint(epoch, avg_loss)

    def _save_checkpoint(self, epoch, loss):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'wm_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.world_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def validate(self):
        print("Validation for World Model not implemented yet.")
        pass
