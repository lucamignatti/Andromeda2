import os
import torch
import torch.optim as optim
import numpy as np
from src.training.base_trainer import BaseTrainer
from src.components.state_encoder import StateEncoder
from src.components.planner import Planner
from src.components.controller import Controller
from src.components.critics import ValueIntrinsicCritic, TemporalC51ExtrinsicCritic
from tests.utils.dummy_env import DummyEnv # Placeholder for actual environment
from src.utils.replay_buffer import ReplayBuffer

class FullAgentTrainer(BaseTrainer):
    """
    Trainer for Stage 3: Full Agent Training.
    """
    def __init__(self):
        super().__init__()
        self.encoder_config = self._load_config('configs/state_encoder.yaml')
        self.planner_config = self._load_config('configs/planner.yaml')
        self.controller_config = self._load_config('configs/controller.yaml')
        self.ext_critic_config = self._load_config('configs/extrinsic_critic.yaml')
        self.int_critic_config = self._load_config('configs/intrinsic_critic.yaml')
        self.train_config = self._load_config('configs/training/stage3_full_agent.yaml')

        # --- Initialize Components ---
        self.state_encoder = StateEncoder(
            input_dim=self.encoder_config['input_dim'],
            output_dim=self.encoder_config['output_dim'],
            hidden_dim=self.encoder_config['hidden_dim']
        ).to(self.device)

        self.planner = Planner(
            encoder_dim=self.encoder_config['output_dim'],
            goal_dim=self.planner_config['goal_dim'],
            mdn_components=self.planner_config['mdn_components'],
            xlstm_config=self.planner_config['xlstm_config']
        ).to(self.device)

        self.controller = Controller(
            encoder_dim=self.encoder_config['output_dim'],
            goal_dim=self.controller_config['goal_dim'],
            hidden_units=self.controller_config['hidden_units'],
            action_dim=self.global_config['action_dim']
        ).to(self.device)

        self.extrinsic_critic = TemporalC51ExtrinsicCritic(
            encoder_dim=self.encoder_config['output_dim'],
            num_atoms=self.ext_critic_config['num_atoms'],
            v_min=self.ext_critic_config['v_min'],
            v_max=self.ext_critic_config['v_max'],
            temporal_horizons=self.ext_critic_config['temporal_horizons'],
            hidden_units=self.ext_critic_config['hidden_units']
        ).to(self.device)

        self.intrinsic_critic = ValueIntrinsicCritic(
            encoder_dim=self.encoder_config['output_dim'],
            goal_dim=self.int_critic_config['goal_dim'],
            hidden_units=self.int_critic_config['hidden_units']
        ).to(self.device)

        # --- Optimizers ---
        self.encoder_optimizer = optim.Adam(self.state_encoder.parameters(), lr=self.train_config['learning_rate_encoder'])
        self.planner_optimizer = optim.Adam(self.planner.parameters(), lr=self.train_config['learning_rate_planner'])
        self.controller_optimizer = optim.Adam(self.controller.parameters(), lr=self.train_config['learning_rate_controller'])
        self.ext_critic_optimizer = optim.Adam(self.extrinsic_critic.parameters(), lr=self.train_config['learning_rate_ext_critic'])
        self.int_critic_optimizer = optim.Adam(self.intrinsic_critic.parameters(), lr=self.train_config['learning_rate_int_critic'])

        self.checkpoint_dir = self._create_checkpoint_dir('stage3_full_agent')
        self.env = DummyEnv(
            observation_shape=tuple(self.global_config['observation_shape']),
            action_dim=self.global_config['action_dim'],
            reward_dim=self.global_config['reward_dim']
        )

    

    def train(self):
        """Main training loop for the full agent."""
        for epoch in range(self.train_config['epochs']):
            self.state_encoder.train()
            self.planner.train()
            self.controller.train()
            self.extrinsic_critic.train()
            self.intrinsic_critic.train()

            # --- Data Collection ---
            observations, actions, rewards, next_observations = [], [], [], []
            current_obs = self.env.reset()
            for _ in range(self.train_config['collect_steps']):
                # Encode the current observation
                encoded_obs = self.state_encoder(torch.from_numpy(current_obs).float().unsqueeze(0).to(self.device))

                # Planner deliberation (simplified for now, will be expanded)
                # For now, let's just pick a random goal from the MDN components
                pi, mu, sigma, _ = self.planner(encoded_obs.unsqueeze(1)) # Planner expects sequence
                # Select a goal based on pi (for now, just pick the first one)
                goal = mu[:, 0, :].squeeze(0) # Take the first component's mean as the goal

                # Controller acts
                action = self.controller(encoded_obs, goal).squeeze(0).cpu().numpy()

                # Step environment
                next_obs, reward, done, _ = self.env.step(action)

                observations.append(current_obs)
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_obs)

                current_obs = next_obs
                if done:
                    current_obs = self.env.reset()
            
            # Add collected data to replay buffer as a single trajectory
            trajectory = list(zip(observations, actions, rewards, next_observations))
            self.replay_buffer.add(trajectory)

            # --- Training Loop (if enough data in buffer) ---
            if len(self.replay_buffer) < self.train_config['batch_size']:
                print(f"Epoch {epoch+1}: Replay buffer not full enough ({len(self.replay_buffer)}/{self.train_config['batch_size']}). Skipping training.")
                continue

            # Sample a batch from the replay buffer
            obs_batch, act_batch, rew_batch, next_obs_batch = self.replay_buffer.sample(self.train_config['batch_size'])

            # Convert to tensors
            obs_batch = torch.from_numpy(obs_batch).float().to(self.device)
            act_batch = torch.from_numpy(act_batch).float().to(self.device)
            rew_batch = torch.from_numpy(rew_batch).float().to(self.device).unsqueeze(-1) # Add dim for reward
            next_obs_batch = torch.from_numpy(next_obs_batch).float().to(self.device)

            # Encode states
            encoded_obs_batch = self.state_encoder(obs_batch)
            encoded_next_obs_batch = self.state_encoder(next_obs_batch)

            # --- Planner and Critic Training ---
            # Planner Loss (simplified for now)
            # For now, we'll just train the planner to output a uniform distribution over goals
            # This will be replaced by a more sophisticated loss later.
            pi, mu, sigma, _ = self.planner(encoded_obs_batch) # Planner expects sequence
            planner_loss = -torch.mean(torch.log(pi + 1e-8)) # Encourage uniform distribution

            # Extrinsic Critic Loss (C51)
            # This is a placeholder. Actual C51 loss is more complex.
            predicted_ext_values = self.extrinsic_critic(encoded_obs_batch)
            # For now, let's just use a simple MSE loss against the rewards
            ext_critic_loss = torch.nn.functional.mse_loss(predicted_ext_values.mean(dim=-1), rew_batch)

            # Intrinsic Critic Loss
            # This requires a goal. For now, let's use a random goal for simplicity.
            random_goals = torch.randn_like(mu[:, 0, :]).unsqueeze(1).repeat(1, encoded_obs_batch.shape[1], 1) # Random goals for each step in sequence
            predicted_int_values = self.intrinsic_critic(encoded_obs_batch, random_goals)
            int_critic_loss = torch.nn.functional.mse_loss(predicted_int_values, rew_batch) # Placeholder

            # Controller Loss
            # This also requires a goal. For now, let's use a random goal for simplicity.
            predicted_actions = self.controller(encoded_obs_batch, random_goals)
            controller_loss_val = torch.nn.functional.mse_loss(predicted_actions, act_batch) # Placeholder

            # --- Combined Optimization ---
            self.encoder_optimizer.zero_grad()
            self.planner_optimizer.zero_grad()
            self.controller_optimizer.zero_grad()
            self.ext_critic_optimizer.zero_grad()
            self.int_critic_optimizer.zero_grad()

            total_loss = planner_loss + ext_critic_loss + int_critic_loss + controller_loss_val
            total_loss.backward()

            self.encoder_optimizer.step()
            self.planner_optimizer.step()
            self.controller_optimizer.step()
            self.ext_critic_optimizer.step()
            self.int_critic_optimizer.step()

            print(f"Epoch {epoch+1}/{self.train_config['epochs']}, Total Loss: {total_loss.item():.4f}")

            if (epoch + 1) % self.train_config['checkpoint_interval'] == 0:
                self._save_checkpoint(epoch, total_loss.item())

    def _save_checkpoint(self, epoch, loss):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'full_agent_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'state_encoder_state_dict': self.state_encoder.state_dict(),
            'planner_state_dict': self.planner.state_dict(),
            'controller_state_dict': self.controller.state_dict(),
            'extrinsic_critic_state_dict': self.extrinsic_critic.state_dict(),
            'intrinsic_critic_state_dict': self.intrinsic_critic.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'planner_optimizer_state_dict': self.planner_optimizer.state_dict(),
            'controller_optimizer_state_dict': self.controller_optimizer.state_dict(),
            'ext_critic_optimizer_state_dict': self.ext_critic_optimizer.state_dict(),
            'int_critic_optimizer_state_dict': self.int_critic_optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def validate(self):
        print("Validation for Full Agent not implemented yet.")
        pass
