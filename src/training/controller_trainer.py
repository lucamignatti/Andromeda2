import os
import torch
import torch.optim as optim
from src.training.base_trainer import BaseTrainer
from src.components.state_encoder import StateEncoder
from src.components.controller import Controller
from src.components.critics import ValueIntrinsicCritic
from tests.utils.dummy_env import DummyEnv

class ControllerTrainer(BaseTrainer):
    """
    Trainer for Stage 2: Controller Pre-training.
    """
    def __init__(self):
        super().__init__()
        self.ctrl_config = self._load_config('configs/controller.yaml')
        self.train_config = self._load_config('configs/training/stage2_controller.yaml')
        self.encoder_config = self._load_config('configs/state_encoder.yaml')
        self.critic_config = self._load_config('configs/intrinsic_critic.yaml')

        # --- Initialize State Encoder ---
        self.state_encoder = StateEncoder(
            input_dim=self.encoder_config['input_dim'],
            output_dim=self.encoder_config['output_dim'],
            hidden_dim=self.encoder_config['hidden_dim']
        ).to(self.device)

        # --- Initialize Controller and Critic ---
        self.controller = Controller(
            encoder_dim=self.encoder_config['output_dim'],
            goal_dim=self.ctrl_config['goal_dim'],
            hidden_units=self.ctrl_config['hidden_units'],
            action_dim=self.global_config['action_dim']
        ).to(self.device)

        self.intrinsic_critic = ValueIntrinsicCritic(
            encoder_dim=self.encoder_config['output_dim'],
            goal_dim=self.ctrl_config['goal_dim'],
            hidden_units=self.critic_config['hidden_units']
        ).to(self.device)

        # --- Optimizers ---
        self.controller_optimizer = optim.Adam(self.controller.parameters(), lr=self.train_config['learning_rate_controller'])
        self.critic_optimizer = optim.Adam(self.intrinsic_critic.parameters(), lr=self.train_config['learning_rate_critic'])
        self.state_encoder_optimizer = optim.Adam(self.state_encoder.parameters(), lr=self.train_config['learning_rate_critic']) # Using critic LR for encoder for now

        self.checkpoint_dir = self._create_checkpoint_dir('stage2_controller')
        self.env = DummyEnv(
            observation_shape=tuple(self.global_config['observation_shape']),
            action_dim=self.global_config['action_dim'],
            reward_dim=self.global_config['reward_dim']
        )

    def train(self):
        """Main training loop for the controller."""
        for epoch in range(self.train_config['epochs']):
            self.controller.train()
            self.intrinsic_critic.train()
            self.state_encoder.train()

            # --- Generate a Goal and Initial State ---
            # For pre-training, we generate random goals and initial states
            initial_state = torch.randn(1, self.encoder_config['input_dim']).to(self.device)
            encoded_initial_state = self.state_encoder(initial_state)
            
            # Generate a random goal for the controller to aim for
            goal_latent = torch.randn(1, self.ctrl_config['goal_dim']).to(self.device)

            # --- Train Controller and Critic ---
            current_encoded_state = encoded_initial_state
            total_controller_loss = 0
            total_critic_loss = 0

            for t in range(self.train_config['train_horizon']):
                # Controller takes current encoded state and goal, and produces an action
                action = self.controller(current_encoded_state, goal_latent)

                # Simulate next state (for now, just a random change for pre-training)
                # In a real scenario, this would come from the environment
                next_state_raw = initial_state + torch.randn_like(initial_state) * 0.1 # Small random perturbation
                next_encoded_state = self.state_encoder(next_state_raw)

                # Critic evaluates how well the controller is doing
                predicted_value = self.intrinsic_critic(current_encoded_state, goal_latent)

                # Calculate intrinsic reward (how much closer we got to the goal)
                # A simple reward is the negative distance to the distance to the goal
                intrinsic_reward = -torch.norm(next_encoded_state - goal_latent, p=2, dim=-1).detach()
                
                # Calculate target for the critic
                with torch.no_grad():
                    next_value = self.intrinsic_critic(next_encoded_state, goal_latent)
                    target_value = intrinsic_reward + self.train_config['gamma'] * next_value

                # --- Combined Update ---
                # Zero out gradients for all optimizers before the backward pass
                self.controller_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.state_encoder_optimizer.zero_grad()

                # Calculate the two loss components
                critic_loss = torch.nn.functional.mse_loss(predicted_value, target_value)
                # The controller loss should encourage actions that lead to higher intrinsic rewards
                ctrl_loss = -(predicted_value * intrinsic_reward).mean()
                
                # Combine the losses
                total_loss = critic_loss + ctrl_loss

                # Backpropagate the total loss. Gradients will be computed for the controller,
                # the critic, and the state encoder's parameters.
                total_loss.backward()

                # Step all optimizers to update their respective network weights
                self.controller_optimizer.step()
                self.critic_optimizer.step()
                self.state_encoder_optimizer.step()

                total_controller_loss += ctrl_loss.item()
                total_critic_loss += critic_loss.item()
                
                # Detach the next_encoded_state from the graph before the next iteration
                current_encoded_state = next_encoded_state.detach()

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
            'state_encoder_state_dict': self.state_encoder.state_dict(),
            'controller_optimizer_state_dict': self.controller_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'state_encoder_optimizer_state_dict': self.state_encoder_optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def validate(self):
        print("Validation for Controller not implemented yet.")
        pass
