import os
import torch
import torch.optim as optim
import numpy as np
from src.training.base_trainer import BaseTrainer
from src.components.world_model import WorldModel
from src.components.controller import Controller
from src.components.critics import ValueIntrinsicCritic
from src.utils.losses import controller_loss
from tests.utils.dummy_env import DummyEnv

class ControllerTrainer(BaseTrainer):
    """
    Trainer for Stage 2: Controller Pre-training.
    """
    def __init__(self, world_model_checkpoint_path):
        super().__init__()
        self.ctrl_config = self._load_config('configs/controller.yaml')
        self.train_config = self._load_config('configs/training/stage2_controller.yaml')
        self.wm_config = self._load_config('configs/world_model.yaml')
        self.critic_config = self._load_config('configs/intrinsic_critic.yaml')

        # --- Load Pre-trained World Model ---
        self.world_model = self._load_world_model(world_model_checkpoint_path)

        # --- Initialize Controller and Critic ---
        self.controller = Controller(
            state_dim=self.wm_config['latent_dim'],
            goal_dim=self.wm_config['latent_dim'],
            hidden_units=self.ctrl_config['hidden_units'],
            action_dim=self.global_config['action_dim']
        ).to(self.device)

        self.intrinsic_critic = ValueIntrinsicCritic(
            latent_dim=self.wm_config['latent_dim'],
            goal_dim=self.wm_config['latent_dim'],
            hidden_units=self.critic_config['hidden_units']
        ).to(self.device)

        # --- Optimizers ---
        self.controller_optimizer = optim.Adam(self.controller.parameters(), lr=self.train_config['learning_rate_controller'])
        self.critic_optimizer = optim.Adam(self.intrinsic_critic.parameters(), lr=self.train_config['learning_rate_critic'])

        self.checkpoint_dir = self._create_checkpoint_dir('stage2_controller')
        self.env = DummyEnv(
            observation_shape=tuple(self.global_config['observation_shape']),
            action_dim=self.global_config['action_dim'],
            reward_dim=self.global_config['reward_dim']
        )

    def _load_world_model(self, checkpoint_path):
        """Loads a pre-trained world model from a checkpoint."""
        world_model = WorldModel(
            observation_shape=tuple(self.global_config['observation_shape']),
            action_dim=self.global_config['action_dim'],
            latent_dim=self.wm_config['latent_dim'],
            reward_dim=self.global_config['reward_dim'],
            xlstm_config=self.wm_config['xlstm_config']
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        world_model.load_state_dict(checkpoint['model_state_dict'])
        world_model.eval() # Set to evaluation mode
        print(f"Loaded pre-trained World Model from {checkpoint_path}")
        return world_model

    def train(self):
        """Main training loop for the controller."""
        for epoch in range(self.train_config['epochs']):
            self.controller.train()
            self.intrinsic_critic.train()

            # --- Generate a Goal and Initial State ---
            # In a real scenario, this would come from the Planner or a curriculum
            # Here, we simulate it by running the world model for a few steps
            initial_obs = torch.from_numpy(self.env.reset()).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Encode the initial observation to get the starting latent state
                _, _, _, _, _, _, start_latent, _ = self.world_model(initial_obs, torch.zeros(1, self.global_config['action_dim']).to(self.device), torch.zeros(1, self.wm_config['latent_dim']).to(self.device))
                
                # Generate a goal by imagining a few steps into the future with random actions
                goal_latent = start_latent
                hidden_state = None
                for _ in range(self.train_config['goal_horizon']):
                    random_action = torch.randn(1, self.global_config['action_dim']).to(self.device)
                    goal_latent, _, _, _, _, _, _, hidden_state = self.world_model(
                        initial_obs, random_action, goal_latent, hidden_state
                    )

            # --- Train Controller and Critic ---
            current_latent = start_latent
            total_controller_loss = 0
            total_critic_loss = 0

            for t in range(self.train_config['train_horizon']):
                # Controller takes current state and goal, and produces an action
                action = self.controller(current_latent, goal_latent)

                # Use the World Model to get the next latent state
                # The world_model is in eval() mode, so its weights are frozen.
                # We run it here to get the next_latent, which is part of the computation graph for the controller.
                next_latent, _, _, _, _, _, _, _ = self.world_model(
                    initial_obs, action, current_latent
                )

                # Critic evaluates how well the controller is doing
                predicted_value = self.intrinsic_critic(current_latent, goal_latent)

                # Calculate intrinsic reward (how much closer we got to the goal)
                # A simple reward is the negative distance to the goal
                intrinsic_reward = -torch.norm(next_latent - goal_latent, p=2, dim=-1).detach()
                
                # Calculate target for the critic
                with torch.no_grad():
                    next_value = self.intrinsic_critic(next_latent, goal_latent)
                    target_value = intrinsic_reward + self.train_config['gamma'] * next_value

                # --- Combined Update ---
                # Zero out gradients for both optimizers before the backward pass
                self.controller_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Calculate the two loss components
                critic_loss = torch.nn.functional.mse_loss(predicted_value, target_value)
                ctrl_loss = -self.intrinsic_critic(next_latent, goal_latent).mean()
                
                # Combine the losses
                total_loss = critic_loss + ctrl_loss

                # Backpropagate the total loss. Gradients will be computed for both
                # the controller and the critic's parameters.
                total_loss.backward()

                # Step both optimizers to update their respective network weights
                self.controller_optimizer.step()
                self.critic_optimizer.step()

                total_controller_loss += ctrl_loss.item()
                total_critic_loss += critic_loss.item()
                
                # Detach the next_latent from the graph before the next iteration
                current_latent = next_latent.detach()

            avg_ctrl_loss = total_controller_loss / self.train_config['train_horizon']
            avg_critic_loss = total_critic_loss / self.train_config['train_horizon']
            
            print(f"Epoch {epoch+1}/{self.train_config['epochs']}, Controller Loss: {avg_ctrl_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")

            if (epoch + 1) % self.train_config['checkpoint_interval'] == 0:
                self._save_checkpoint(epoch, avg_ctrl_loss)

    def _save_checkpoint(self, epoch, loss):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'ctrl_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'controller_state_dict': self.controller.state_dict(),
            'critic_state_dict': self.intrinsic_critic.state_dict(),
            'controller_optimizer_state_dict': self.controller_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def validate(self):
        print("Validation for Controller not implemented yet.")
        pass
