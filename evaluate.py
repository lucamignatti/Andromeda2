#!/usr/bin/env python3
"""
Evaluation script for trained Andromeda2 models.
This script loads trained models and evaluates their performance against various opponents.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from typing import Dict, Any, List, Optional
import time
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import Andromeda2 components
from src.models.agent import Andromeda2Agent
from src.environments.factory import make_env
from src.utils.metrics import TrainingMetrics

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available, plotting disabled")


class ModelEvaluator:
    """Comprehensive model evaluation suite."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = 'auto'
    ):
        """
        Initialize model evaluator.

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to training configuration
            device: Device to use for evaluation
        """
        self.model_path = model_path
        self.config_path = config_path

        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load model
        self.agent = self._load_model()

        # Evaluation metrics
        self.metrics = defaultdict(list)

    def _load_model(self) -> Andromeda2Agent:
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Create agent
        agent_config = self.config['agent']
        planner_config = self.config['planner']
        controller_config = self.config['controller']

        agent = Andromeda2Agent(
            observation_size=agent_config['observation_size'],
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

        # Load state dict
        if 'model_state_dict' in checkpoint:
            agent.load_state_dict(checkpoint['model_state_dict'])
        else:
            agent.load_state_dict(checkpoint)

        agent.to(self.device)
        agent.eval()

        print(f"Model loaded successfully on {self.device}")
        return agent

    def evaluate_episodes(
        self,
        env_type: str = "1v1",
        n_episodes: int = 100,
        max_steps: int = 10000,
        deterministic: bool = True,
        render: bool = False,
        save_replays: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate agent over multiple episodes.

        Args:
            env_type: Type of environment to evaluate on
            n_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            deterministic: Whether to use deterministic actions
            render: Whether to render episodes
            save_replays: Whether to save episode replays

        Returns:
            Dictionary of evaluation results
        """
        print(f"Evaluating {n_episodes} episodes on {env_type} environment...")

        # Create environment
        env = make_env(
            env_type=env_type,
            hierarchical=self.config['environment']['hierarchical'],
            tick_skip=self.config['environment']['tick_skip'],
            game_speed=self.config['environment']['game_speed'],
            spawn_opponents=self.config['environment']['spawn_opponents'],
            render=render
        )

        episode_rewards = []
        episode_lengths = []
        episode_goals_scored = []
        episode_goals_conceded = []
        episode_times = []
        goal_vectors_history = []

        for episode in range(n_episodes):
            print(f"Episode {episode + 1}/{n_episodes}", end='\r')

            # Reset environment and agent
            obs = env.reset()
            self.agent.reset_episode(1, self.device)

            total_reward = 0
            episode_length = 0
            goals_scored = 0
            goals_conceded = 0
            episode_goal_vectors = []

            start_time = time.time()
            done = False

            while not done and episode_length < max_steps:
                # Get agent action
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                    # Get action and goal vector
                    action = self.agent.get_action(obs_tensor, deterministic=deterministic)
                    goal_vector = self.agent.get_goal_vector(obs_tensor)

                    # Store goal vector
                    episode_goal_vectors.append(goal_vector.cpu().numpy())

                    # Convert action to numpy
                    if isinstance(action, torch.Tensor):
                        action_np = action.cpu().numpy()
                        if action_np.ndim > 1:
                            action_np = action_np.squeeze(0)
                    else:
                        action_np = action

                # Step environment
                next_obs, reward, done, info = env.step(action_np)

                total_reward += reward
                episode_length += 1

                # Track goals (if available in info)
                if isinstance(info, dict):
                    if info.get('goal_scored', False):
                        goals_scored += 1
                    if info.get('goal_conceded', False):
                        goals_conceded += 1

                obs = next_obs

            episode_time = time.time() - start_time

            # Store episode data
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            episode_goals_scored.append(goals_scored)
            episode_goals_conceded.append(goals_conceded)
            episode_times.append(episode_time)
            goal_vectors_history.append(np.array(episode_goal_vectors))

        print(f"\nCompleted {n_episodes} episodes")

        # Calculate statistics
        results = {
            'n_episodes': n_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_goals_scored': np.mean(episode_goals_scored),
            'mean_goals_conceded': np.mean(episode_goals_conceded),
            'win_rate': np.mean([r > 0 for r in episode_rewards]),
            'mean_episode_time': np.mean(episode_times),
            'total_time': np.sum(episode_times),
            'fps': np.sum(episode_lengths) / np.sum(episode_times)
        }

        # Store detailed data
        results['episode_rewards'] = episode_rewards
        results['episode_lengths'] = episode_lengths
        results['goal_vectors_history'] = goal_vectors_history

        # Close environment
        if hasattr(env, 'close'):
            env.close()

        return results

    def analyze_goal_vectors(
        self,
        goal_vectors_history: List[np.ndarray],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze goal vector evolution and patterns.

        Args:
            goal_vectors_history: List of goal vector sequences from episodes
            save_path: Path to save analysis plots

        Returns:
            Dictionary of analysis results
        """
        print("Analyzing goal vector patterns...")

        if not goal_vectors_history:
            return {}

        # Concatenate all goal vectors
        all_goal_vectors = np.concatenate(goal_vectors_history, axis=0)

        # Basic statistics
        analysis = {
            'mean_values': np.mean(all_goal_vectors, axis=0),
            'std_values': np.std(all_goal_vectors, axis=0),
            'min_values': np.min(all_goal_vectors, axis=0),
            'max_values': np.max(all_goal_vectors, axis=0),
            'mean_norm': np.mean(np.linalg.norm(all_goal_vectors, axis=1)),
            'std_norm': np.std(np.linalg.norm(all_goal_vectors, axis=1))
        }

        # Component names for 12D goal vector
        component_names = [
            "Car Vel X", "Car Vel Y", "Car Vel Z",
            "Ball Vel X", "Ball Vel Y", "Ball Vel Z",
            "Car-Ball X", "Car-Ball Y", "Car-Ball Z",
            "Ball-Goal X", "Ball-Goal Y", "Ball-Goal Z"
        ]

        # Calculate component usage (how often each component is significantly non-zero)
        threshold = 0.1
        component_usage = np.mean(np.abs(all_goal_vectors) > threshold, axis=0)
        analysis['component_usage'] = dict(zip(component_names, component_usage))

        # Calculate goal vector stability (how much it changes between steps)
        stability_scores = []
        for episode_goals in goal_vectors_history:
            if len(episode_goals) > 1:
                changes = np.diff(episode_goals, axis=0)
                stability = 1.0 / (1.0 + np.mean(np.linalg.norm(changes, axis=1)))
                stability_scores.append(stability)

        analysis['mean_stability'] = np.mean(stability_scores) if stability_scores else 0.0

        # Plot analysis if matplotlib is available
        if MATPLOTLIB_AVAILABLE and save_path:
            self._plot_goal_vector_analysis(all_goal_vectors, component_names, save_path)

        return analysis

    def _plot_goal_vector_analysis(
        self,
        goal_vectors: np.ndarray,
        component_names: List[str],
        save_path: str
    ):
        """Plot goal vector analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Distribution of goal vector norms
        axes[0, 0].hist(np.linalg.norm(goal_vectors, axis=1), bins=50, alpha=0.7)
        axes[0, 0].set_title('Distribution of Goal Vector Norms')
        axes[0, 0].set_xlabel('L2 Norm')
        axes[0, 0].set_ylabel('Frequency')

        # Component statistics
        means = np.mean(goal_vectors, axis=0)
        stds = np.std(goal_vectors, axis=0)

        x_pos = np.arange(len(component_names))
        axes[0, 1].bar(x_pos, means, yerr=stds, alpha=0.7)
        axes[0, 1].set_title('Goal Vector Component Statistics')
        axes[0, 1].set_xlabel('Component')
        axes[0, 1].set_ylabel('Mean Value')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(component_names, rotation=45, ha='right')

        # Correlation matrix
        corr_matrix = np.corrcoef(goal_vectors.T)
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Goal Vector Component Correlations')
        axes[1, 0].set_xticks(range(len(component_names)))
        axes[1, 0].set_yticks(range(len(component_names)))
        axes[1, 0].set_xticklabels(component_names, rotation=45, ha='right')
        axes[1, 0].set_yticklabels(component_names)
        plt.colorbar(im, ax=axes[1, 0])

        # Time series of first few components
        if len(goal_vectors) > 100:
            sample_indices = np.linspace(0, len(goal_vectors)-1, 500, dtype=int)
            sample_goals = goal_vectors[sample_indices]
        else:
            sample_goals = goal_vectors

        for i in range(min(4, goal_vectors.shape[1])):
            axes[1, 1].plot(sample_goals[:, i], label=component_names[i], alpha=0.7)
        axes[1, 1].set_title('Goal Vector Evolution (Sample)')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Goal Value')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def compare_with_baseline(
        self,
        baseline_results: Dict[str, Any],
        current_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare current model with baseline results.

        Args:
            baseline_results: Results from baseline model
            current_results: Results from current model

        Returns:
            Comparison metrics
        """
        comparison = {}

        metrics_to_compare = [
            'mean_reward', 'win_rate', 'mean_goals_scored',
            'mean_goals_conceded', 'mean_length', 'fps'
        ]

        for metric in metrics_to_compare:
            if metric in baseline_results and metric in current_results:
                baseline_val = baseline_results[metric]
                current_val = current_results[metric]

                improvement = current_val - baseline_val
                improvement_pct = (improvement / baseline_val * 100) if baseline_val != 0 else 0

                comparison[f'{metric}_improvement'] = improvement
                comparison[f'{metric}_improvement_pct'] = improvement_pct

        return comparison

    def generate_report(
        self,
        results: Dict[str, Any],
        analysis: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate evaluation report.

        Args:
            results: Evaluation results
            analysis: Goal vector analysis
            save_path: Path to save report

        Returns:
            Report text
        """
        report_lines = [
            "=" * 60,
            "ANDROMEDA2 MODEL EVALUATION REPORT",
            "=" * 60,
            "",
            f"Model: {self.model_path}",
            f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Device: {self.device}",
            "",
            "PERFORMANCE METRICS:",
            "-" * 30,
            f"Episodes Evaluated: {results['n_episodes']}",
            f"Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}",
            f"Win Rate: {results['win_rate']:.2%}",
            f"Mean Episode Length: {results['mean_length']:.1f} steps",
            f"Mean Goals Scored: {results['mean_goals_scored']:.2f}",
            f"Mean Goals Conceded: {results['mean_goals_conceded']:.2f}",
            f"Evaluation FPS: {results['fps']:.1f}",
            "",
            "GOAL VECTOR ANALYSIS:",
            "-" * 30,
            f"Mean Goal Vector Norm: {analysis.get('mean_norm', 0):.4f}",
            f"Goal Vector Stability: {analysis.get('mean_stability', 0):.4f}",
            ""
        ]

        # Component usage
        if 'component_usage' in analysis:
            report_lines.extend([
                "Component Usage (% of time significantly active):",
                ""
            ])
            for component, usage in analysis['component_usage'].items():
                report_lines.append(f"  {component}: {usage:.1%}")
            report_lines.append("")

        # Performance distribution
        report_lines.extend([
            "PERFORMANCE DISTRIBUTION:",
            "-" * 30,
            f"Best Episode: {results['max_reward']:.4f}",
            f"Worst Episode: {results['min_reward']:.4f}",
            f"Median Reward: {np.median(results['episode_rewards']):.4f}",
            ""
        ])

        report_text = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")

        return report_text


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained Andromeda2 model')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to evaluate')
    parser.add_argument('--env-type', type=str, default='1v1',
                       choices=['1v1', '2v2', '3v3', 'training'],
                       help='Environment type for evaluation')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic actions')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for evaluation')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline results for comparison')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.config, args.device)

    # Run evaluation
    print(f"Starting evaluation of model: {args.model_path}")
    results = evaluator.evaluate_episodes(
        env_type=args.env_type,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render
    )

    # Analyze goal vectors
    analysis = evaluator.analyze_goal_vectors(
        results['goal_vectors_history'],
        save_path=os.path.join(args.output_dir, 'goal_vector_analysis.png')
    )

    # Generate report
    report = evaluator.generate_report(
        results,
        analysis,
        save_path=os.path.join(args.output_dir, 'evaluation_report.txt')
    )

    print(report)

    # Save detailed results
    results_file = os.path.join(args.output_dir, 'detailed_results.npz')
    np.savez(
        results_file,
        episode_rewards=results['episode_rewards'],
        episode_lengths=results['episode_lengths'],
        goal_vectors_history=results['goal_vectors_history'],
        analysis=analysis
    )
    print(f"Detailed results saved to {results_file}")

    # Compare with baseline if provided
    if args.baseline:
        try:
            baseline_data = np.load(args.baseline, allow_pickle=True)
            baseline_results = baseline_data['results'].item()

            comparison = evaluator.compare_with_baseline(baseline_results, results)

            print("\nBASELINE COMPARISON:")
            print("-" * 30)
            for metric, value in comparison.items():
                if 'improvement_pct' in metric:
                    print(f"{metric}: {value:.2f}%")
                else:
                    print(f"{metric}: {value:.4f}")

        except Exception as e:
            print(f"Warning: Could not load baseline results: {e}")

    print(f"\nEvaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
