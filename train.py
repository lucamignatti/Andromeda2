#!/usr/bin/env python3
"""
Main training script for Andromeda2 Hierarchical RL Agent.
This script orchestrates the training of the strategic planner and motor controller.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
from typing import Dict, Any, Optional
import warnings
import traceback
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import Andromeda2 components
from src.models.agent import Andromeda2Agent
from src.training.ppo_hierarchical import HierarchicalPPOTrainer
from src.environments.vectorized import make_vectorized_env
from src.environments.factory import make_env
from src.utils.metrics import TrainingMetrics, PerformanceProfiler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Weights & Biases not available")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config: Dict[str, Any]) -> str:
    """Setup and return the appropriate device."""
    device_config = config.get('hardware', {}).get('device', 'auto')

    if device_config == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            print("Using CPU")
    else:
        device = device_config
        print(f"Using device: {device}")

    return device


def setup_directories(config: Dict[str, Any]):
    """Create necessary directories."""
    log_dir = config.get('logging', {}).get('log_dir', './logs')
    checkpoint_dir = config.get('logging', {}).get('checkpoint_dir', './checkpoints')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    return log_dir, checkpoint_dir


def setup_wandb(config: Dict[str, Any], run_name: str):
    """Setup Weights & Biases logging."""
    if not WANDB_AVAILABLE or not config.get('logging', {}).get('use_wandb', False):
        return

    wandb_config = config.get('logging', {})

    wandb.init(
        project=wandb_config.get('wandb_project', 'andromeda2'),
        entity=wandb_config.get('wandb_entity', None),
        name=run_name,
        config=config,
        tags=['hierarchical-rl', 'rocket-league', 'xLSTM']
    )


def create_environments(config: Dict[str, Any], device: str):
    """Create training and evaluation environments."""
    env_config = config['environment']

    # Training environment
    train_env = make_vectorized_env(
        num_envs=env_config['num_envs'],
        env_config={
            'env_type': env_config['type'],
            'hierarchical': False,  # Temporarily disable hierarchical wrapper
            'tick_skip': env_config['tick_skip'],
            'game_speed': env_config['game_speed'],
            'spawn_opponents': env_config['spawn_opponents'],
            'auto_detect_demos': env_config['auto_detect_demos'],
            'training_type': env_config.get('training_type', 'basic')
        },
        async_env=env_config.get('async_env', False),
        device=device
    )

    # Evaluation environment (single environment)
    eval_env = make_env(
        env_type=env_config['type'],
        hierarchical=False,  # Temporarily disable hierarchical wrapper
        tick_skip=env_config['tick_skip'],
        game_speed=env_config['game_speed'],
        spawn_opponents=env_config['spawn_opponents'],
        auto_detect_demos=env_config['auto_detect_demos']
    )

    return train_env, eval_env


def create_agent(config: Dict[str, Any], observation_size: int, device: str) -> Andromeda2Agent:
    """Create the Andromeda2 agent."""
    agent_config = config['agent']
    planner_config = config['planner']
    controller_config = config['controller']

    agent = Andromeda2Agent(
        observation_size=observation_size,
        action_size=agent_config['action_size'],
        goal_vector_dim=agent_config['goal_vector_dim'],
        planner_config=planner_config,
        controller_config=controller_config,
        planner_update_freq=agent_config['planner_update_freq'],
        use_memory_replay=agent_config['use_memory_replay'],
        controller_type=agent_config['controller_type'],
        goal_vector_scaling=agent_config['goal_vector_scaling'],
        training_mode=agent_config['training_mode']
    )

    return agent.to(device)


def create_trainer(
    agent: Andromeda2Agent,
    train_env,
    config: Dict[str, Any],
    device: str
) -> HierarchicalPPOTrainer:
    """Create the hierarchical PPO trainer."""
    training_config = config['training']
    intrinsic_config = config['intrinsic_rewards']

    # Convert learning_rate to float if it's a string
    learning_rate = float(training_config['learning_rate'])

    trainer = HierarchicalPPOTrainer(
        agent=agent,
        env=train_env,
        learning_rate=learning_rate,
        gamma=training_config['gamma'],
        gae_lambda=training_config['gae_lambda'],
        clip_range=training_config['clip_range'],
        entropy_coef=training_config['entropy_coef'],
        value_loss_coef=training_config['value_loss_coef'],
        max_grad_norm=training_config['max_grad_norm'],
        epochs_per_update=training_config['epochs_per_update'],
        batch_size=training_config['batch_size'],
        n_steps=training_config['n_steps'],
        intrinsic_reward_weights=intrinsic_config,
        planner_lr_multiplier=training_config['planner_lr_multiplier'],
        controller_lr_multiplier=training_config['controller_lr_multiplier'],
        separate_optimizers=training_config['separate_optimizers'],
        normalize_advantages=training_config['normalize_advantages'],
        normalize_intrinsic_rewards=training_config['normalize_intrinsic_rewards'],
        target_kl=training_config['target_kl'],
        device=device
    )

    return trainer


def training_callback(locals_dict, globals_dict, metrics: TrainingMetrics, config: Dict[str, Any]):
    """Callback function for training loop."""
    trainer = locals_dict['self']

    # Get current stats
    stats = {
        'timesteps': trainer.num_timesteps,
        'updates': trainer.num_updates,
        'fps': metrics.get_fps(),
        'runtime_hours': metrics.get_runtime() / 3600
    }

    # Log to metrics
    metrics.log_step_metrics(stats)

    # Check for early stopping conditions
    if config.get('debug', {}).get('check_numerics', False):
        # Check for NaN/Inf in agent parameters
        for name, param in trainer.agent.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN/Inf detected in parameter {name}")
                return False

    return True


def evaluate_agent(agent: Andromeda2Agent, eval_env, n_episodes: int = 5, device: str = 'cpu'):
    """Evaluate the agent performance."""
    agent.eval()
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs = eval_env.reset()
        agent.reset_episode(1, device)

        total_reward = 0
        episode_length = 0
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = agent.get_action(obs_tensor, deterministic=True)

                # Convert action to numpy for environment
                if isinstance(action, torch.Tensor):
                    action_np = action.cpu().numpy()
                    if action_np.ndim > 1:
                        action_np = action_np.squeeze(0)
                else:
                    action_np = action

            obs, reward, done, info = eval_env.step(action_np)
            total_reward += reward
            episode_length += 1

            # Prevent infinite episodes
            if episode_length > 10000:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Andromeda2 Hierarchical RL Agent')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this training run')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load configuration
    config = load_config(args.config)

    # Setup device
    device = setup_device(config)

    # Setup directories
    log_dir, checkpoint_dir = setup_directories(config)

    # Create run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"andromeda2_{timestamp}"
    else:
        run_name = args.run_name

    # Setup logging
    metrics = TrainingMetrics(
        log_dir=log_dir,
        use_wandb=config.get('logging', {}).get('use_wandb', False),
        use_tensorboard=config.get('logging', {}).get('use_tensorboard', True),
        window_size=config.get('logging', {}).get('window_size', 100)
    )

    # Setup W&B
    setup_wandb(config, run_name)

    # Performance profiler
    profiler = PerformanceProfiler() if config.get('debug', {}).get('profile_performance', False) else None

    try:
        # Create environments
        print("Creating environments...")
        if profiler:
            profiler.start_timer('env_creation')

        train_env, eval_env = create_environments(config, device)

        if profiler:
            profiler.end_timer('env_creation')

        # Get observation size from environment
        dummy_obs = train_env.reset()
        if isinstance(dummy_obs, dict):
            observation_size = dummy_obs['observations'].shape[-1]
        elif isinstance(dummy_obs, list):
            observation_size = dummy_obs[0].shape[-1] if dummy_obs else config['agent']['observation_size']
        else:
            observation_size = dummy_obs.shape[-1]

        print(f"Observation size: {observation_size}")

        # Create agent
        print("Creating agent...")
        if profiler:
            profiler.start_timer('agent_creation')

        agent = create_agent(config, observation_size, device)

        if profiler:
            profiler.end_timer('agent_creation')

        # Print agent info
        total_params = sum(p.numel() for p in agent.parameters())
        trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        print(f"Agent created with {total_params:,} total parameters ({trainable_params:,} trainable)")

        # Load checkpoint if resuming
        if args.resume:
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            agent.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully")

        # Evaluation mode
        if args.eval_only:
            print("Running evaluation...")
            eval_results = evaluate_agent(agent, eval_env,
                                        config.get('logging', {}).get('n_eval_episodes', 5),
                                        device)
            print("Evaluation Results:")
            for key, value in eval_results.items():
                print(f"  {key}: {value:.4f}")
            return

        # Create trainer
        print("Creating trainer...")
        if profiler:
            profiler.start_timer('trainer_creation')

        trainer = create_trainer(agent, train_env, config, device)

        if profiler:
            profiler.end_timer('trainer_creation')

        # Load trainer state if resuming
        if args.resume:
            trainer_checkpoint_path = args.resume.replace('_agent.pt', '_trainer.pt')
            if os.path.exists(trainer_checkpoint_path):
                print(f"Loading trainer state from {trainer_checkpoint_path}")
                trainer.load(trainer_checkpoint_path)

        # Training parameters
        total_timesteps = config['training']['total_timesteps']
        log_interval = config.get('logging', {}).get('log_interval', 1)
        eval_freq = config.get('logging', {}).get('eval_freq', 100)
        save_freq = config.get('logging', {}).get('save_frequency', 1000)

        print(f"Starting training for {total_timesteps:,} timesteps...")
        print(f"Training configuration:")
        print(f"  - Environment: {config['environment']['type']} with {config['environment']['num_envs']} parallel envs")
        print(f"  - Agent: {config['agent']['training_mode']} mode")
        print(f"  - Learning rate: {config['training']['learning_rate']}")
        print(f"  - Batch size: {config['training']['batch_size']}")
        print(f"  - Device: {device}")

        # Define callback with metrics
        def callback(locals_dict, globals_dict):
            return training_callback(locals_dict, globals_dict, metrics, config)

        # Training loop
        trainer.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=config.get('logging', {}).get('n_eval_episodes', 5),
            tb_log_name=run_name
        )

        # Save final model
        final_checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_final.pt")
        agent.save_checkpoint(final_checkpoint_path)
        print(f"Final model saved to {final_checkpoint_path}")

        # Save final trainer state
        final_trainer_path = os.path.join(checkpoint_dir, f"{run_name}_trainer_final.pt")
        trainer.save(final_trainer_path)
        print(f"Final trainer state saved to {final_trainer_path}")

        # Final evaluation
        print("Running final evaluation...")
        final_eval_results = evaluate_agent(agent, eval_env, 10, device)
        print("Final Evaluation Results:")
        for key, value in final_eval_results.items():
            print(f"  {key}: {value:.4f}")

        # Log final results
        metrics.log_episode_metrics({f"final_{k}": v for k, v in final_eval_results.items()})

        # Performance profiling results
        if profiler:
            print("\nPerformance Profiling Results:")
            perf_stats = profiler.get_stats()
            for timer_name, stats in perf_stats.items():
                print(f"  {timer_name}: {stats['mean']:.4f}s ± {stats['std']:.4f}s "
                      f"(total: {stats['total']:.2f}s, count: {stats['count']})")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save checkpoint on interrupt
        interrupt_checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_interrupted.pt")
        if 'agent' in locals():
            agent.save_checkpoint(interrupt_checkpoint_path)
            print(f"Interrupted model saved to {interrupt_checkpoint_path}")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        traceback.print_exc()

        # Save checkpoint on error
        error_checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_error.pt")
        if 'agent' in locals():
            try:
                agent.save_checkpoint(error_checkpoint_path)
                print(f"Error checkpoint saved to {error_checkpoint_path}")
            except:
                print("Failed to save error checkpoint")

    finally:
        # Cleanup
        print("\nCleaning up...")

        # Close environments
        if 'train_env' in locals():
            train_env.close()
        if 'eval_env' in locals() and hasattr(eval_env, 'close'):
            eval_env.close()

        # Close metrics
        if 'metrics' in locals():
            metrics.close()

        # Finish W&B
        if WANDB_AVAILABLE and config.get('logging', {}).get('use_wandb', False):
            try:
                wandb.finish()
            except:
                pass

        print("Training completed!")


if __name__ == "__main__":
    main()
