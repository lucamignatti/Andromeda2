"""
Training metrics tracking and visualization utilities for Andromeda2.
Handles metric collection, analysis, and logging for hierarchical RL training.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import json
import os
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TrainingMetrics:
    """
    Comprehensive metrics tracking for hierarchical RL training.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        window_size: int = 100,
        save_frequency: int = 1000
    ):
        """
        Initialize training metrics tracker.

        Args:
            log_dir: Directory to save logs and plots
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
            window_size: Window size for moving averages
            save_frequency: How often to save metrics to disk
        """
        self.log_dir = log_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.window_size = window_size
        self.save_frequency = save_frequency

        # Create log directory
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        # Initialize loggers
        self.tb_writer = None
        if self.use_tensorboard and self.log_dir:
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)

        # Metrics storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.step_metrics = defaultdict(list)

        # Moving averages
        self.moving_averages = defaultdict(lambda: deque(maxlen=window_size))

        # Training state
        self.start_time = time.time()
        self.step_count = 0
        self.episode_count = 0
        self.update_count = 0

        # Performance tracking
        self.fps_tracker = deque(maxlen=100)
        self.last_step_time = time.time()

    def log_step_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for a training step.

        Args:
            metrics: Dictionary of metric values
            step: Step number (uses internal counter if None)
        """
        if step is None:
            step = self.step_count

        # Store metrics
        for key, value in metrics.items():
            self.step_metrics[key].append(value)
            self.moving_averages[key].append(value)

        # Log to external services
        if self.use_wandb:
            wandb.log(metrics, step=step)

        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"step/{key}", value, step)

        self.step_count += 1

        # Update FPS
        current_time = time.time()
        if self.last_step_time:
            fps = 1.0 / (current_time - self.last_step_time)
            self.fps_tracker.append(fps)
        self.last_step_time = current_time

        # Save periodically
        if self.step_count % self.save_frequency == 0:
            self.save_metrics()

    def log_episode_metrics(self, metrics: Dict[str, float], episode: Optional[int] = None):
        """
        Log metrics for an episode.

        Args:
            metrics: Dictionary of metric values
            episode: Episode number (uses internal counter if None)
        """
        if episode is None:
            episode = self.episode_count

        # Store metrics
        for key, value in metrics.items():
            self.episode_metrics[key].append(value)

        # Log to external services
        if self.use_wandb:
            episode_metrics = {f"episode/{k}": v for k, v in metrics.items()}
            wandb.log(episode_metrics, step=self.step_count)

        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"episode/{key}", value, episode)

        self.episode_count += 1

    def log_update_metrics(self, metrics: Dict[str, float], update: Optional[int] = None):
        """
        Log metrics for a training update.

        Args:
            metrics: Dictionary of metric values
            update: Update number (uses internal counter if None)
        """
        if update is None:
            update = self.update_count

        # Store metrics
        for key, value in metrics.items():
            self.metrics[key].append(value)

        # Log to external services
        if self.use_wandb:
            update_metrics = {f"update/{k}": v for k, v in metrics.items()}
            wandb.log(update_metrics, step=self.step_count)

        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"update/{key}", value, update)

        self.update_count += 1

    def log_hierarchical_metrics(
        self,
        planner_metrics: Dict[str, float],
        controller_metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log hierarchical-specific metrics.

        Args:
            planner_metrics: Metrics for the planner
            controller_metrics: Metrics for the controller
            step: Step number
        """
        # Prefix metrics
        prefixed_planner = {f"planner/{k}": v for k, v in planner_metrics.items()}
        prefixed_controller = {f"controller/{k}": v for k, v in controller_metrics.items()}

        combined_metrics = {**prefixed_planner, **prefixed_controller}
        self.log_step_metrics(combined_metrics, step)

    def log_goal_vector_analysis(
        self,
        goal_vectors: torch.Tensor,
        analysis_results: Dict[str, Any],
        step: Optional[int] = None
    ):
        """
        Log goal vector analysis metrics.

        Args:
            goal_vectors: Current goal vectors
            analysis_results: Analysis results from agent
            step: Step number
        """
        if step is None:
            step = self.step_count

        # Basic statistics
        goal_stats = {
            "goal_vector/mean_norm": torch.norm(goal_vectors, dim=-1).mean().item(),
            "goal_vector/std_norm": torch.norm(goal_vectors, dim=-1).std().item(),
            "goal_vector/max_component": goal_vectors.abs().max().item(),
            "goal_vector/min_component": goal_vectors.abs().min().item()
        }

        # Component-wise statistics
        for i in range(goal_vectors.size(-1)):
            goal_stats[f"goal_vector/component_{i}_mean"] = goal_vectors[:, i].mean().item()
            goal_stats[f"goal_vector/component_{i}_std"] = goal_vectors[:, i].std().item()

        # Analysis results
        if analysis_results:
            for key, value in analysis_results.items():
                if isinstance(value, (int, float)):
                    goal_stats[f"goal_analysis/{key}"] = value
                elif isinstance(value, np.ndarray) and value.size == 1:
                    goal_stats[f"goal_analysis/{key}"] = value.item()

        self.log_step_metrics(goal_stats, step)

        # Log goal vector histogram
        if self.tb_writer:
            self.tb_writer.add_histogram("goal_vectors", goal_vectors, step)

    def get_moving_average(self, metric_name: str, window: Optional[int] = None) -> float:
        """
        Get moving average of a metric.

        Args:
            metric_name: Name of the metric
            window: Window size (uses default if None)

        Returns:
            Moving average value
        """
        if metric_name not in self.moving_averages:
            return 0.0

        values = list(self.moving_averages[metric_name])
        if not values:
            return 0.0

        if window is not None:
            values = values[-window:]

        return np.mean(values)

    def get_fps(self) -> float:
        """Get current training FPS."""
        if not self.fps_tracker:
            return 0.0
        return np.mean(self.fps_tracker)

    def get_runtime(self) -> float:
        """Get total training runtime in seconds."""
        return time.time() - self.start_time

    def get_summary(self) -> Dict[str, Any]:
        """
        Get training summary.

        Returns:
            Dictionary with training summary
        """
        runtime = self.get_runtime()
        fps = self.get_fps()

        summary = {
            "runtime_seconds": runtime,
            "runtime_hours": runtime / 3600,
            "total_steps": self.step_count,
            "total_episodes": self.episode_count,
            "total_updates": self.update_count,
            "fps": fps,
            "steps_per_second": self.step_count / runtime if runtime > 0 else 0
        }

        # Add recent performance metrics
        recent_metrics = {}
        for metric_name in ["planner_loss", "controller_loss", "episode_reward"]:
            if metric_name in self.moving_averages:
                recent_metrics[f"recent_{metric_name}"] = self.get_moving_average(metric_name)

        summary.update(recent_metrics)

        return summary

    def plot_training_curves(
        self,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot training curves.

        Args:
            metrics: List of metrics to plot (plots all if None)
            save_path: Path to save plot
            show: Whether to show plot
        """
        if metrics is None:
            # Plot common metrics
            metrics = ["planner_loss", "controller_loss", "episode_reward", "fps"]

        # Filter available metrics
        available_metrics = [m for m in metrics if m in self.step_metrics or m in self.episode_metrics]

        if not available_metrics:
            warnings.warn("No metrics available for plotting")
            return

        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 4 * len(available_metrics)))
        if len(available_metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            ax = axes[i]

            # Get data
            if metric in self.step_metrics:
                data = self.step_metrics[metric]
                x = range(len(data))
                xlabel = "Steps"
            elif metric in self.episode_metrics:
                data = self.episode_metrics[metric]
                x = range(len(data))
                xlabel = "Episodes"
            else:
                continue

            # Plot
            ax.plot(x, data, alpha=0.3, color='blue', linewidth=0.5)

            # Moving average
            if len(data) > self.window_size:
                moving_avg = [np.mean(data[max(0, i-self.window_size):i+1]) for i in range(len(data))]
                ax.plot(x, moving_avg, color='red', linewidth=2, label='Moving Average')
                ax.legend()

            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_goal_vector_evolution(
        self,
        goal_vectors_history: List[torch.Tensor],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot goal vector evolution over time.

        Args:
            goal_vectors_history: List of goal vectors over time
            save_path: Path to save plot
            show: Whether to show plot
        """
        if not goal_vectors_history:
            return

        # Convert to numpy
        goal_vectors = torch.stack(goal_vectors_history).cpu().numpy()
        n_steps, batch_size, goal_dim = goal_vectors.shape

        # Take first environment for visualization
        goal_vectors = goal_vectors[:, 0, :]

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        component_names = [
            "Car Vel X", "Car Vel Y", "Car Vel Z",
            "Ball Vel X", "Ball Vel Y", "Ball Vel Z",
            "Car-Ball X", "Car-Ball Y", "Car-Ball Z",
            "Ball-Goal X", "Ball-Goal Y", "Ball-Goal Z"
        ]

        for i in range(min(goal_dim, 12)):
            ax = axes[i]
            ax.plot(goal_vectors[:, i])
            ax.set_title(component_names[i])
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Goal Value")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_reward_comparison(
        self,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot comparison between extrinsic and intrinsic rewards.

        Args:
            save_path: Path to save plot
            show: Whether to show plot
        """
        extrinsic_key = "rewards_extrinsic"
        intrinsic_key = "rewards_intrinsic"

        if extrinsic_key not in self.step_metrics or intrinsic_key not in self.step_metrics:
            warnings.warn("Reward data not available for plotting")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Extrinsic rewards
        ax1 = axes[0]
        extrinsic_data = self.step_metrics[extrinsic_key]
        ax1.plot(extrinsic_data, alpha=0.3, color='blue', linewidth=0.5)
        if len(extrinsic_data) > self.window_size:
            moving_avg = [np.mean(extrinsic_data[max(0, i-self.window_size):i+1]) for i in range(len(extrinsic_data))]
            ax1.plot(moving_avg, color='red', linewidth=2, label='Moving Average')
            ax1.legend()
        ax1.set_title("Extrinsic Rewards (Planner)")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Reward")
        ax1.grid(True, alpha=0.3)

        # Intrinsic rewards
        ax2 = axes[1]
        intrinsic_data = self.step_metrics[intrinsic_key]
        ax2.plot(intrinsic_data, alpha=0.3, color='green', linewidth=0.5)
        if len(intrinsic_data) > self.window_size:
            moving_avg = [np.mean(intrinsic_data[max(0, i-self.window_size):i+1]) for i in range(len(intrinsic_data))]
            ax2.plot(moving_avg, color='orange', linewidth=2, label='Moving Average')
            ax2.legend()
        ax2.set_title("Intrinsic Rewards (Controller)")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Reward")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def save_metrics(self, filepath: Optional[str] = None):
        """
        Save metrics to file.

        Args:
            filepath: Path to save metrics (uses default if None)
        """
        if filepath is None and self.log_dir:
            filepath = os.path.join(self.log_dir, "metrics.json")
        elif filepath is None:
            return

        # Convert metrics to serializable format
        serializable_metrics = {}

        for category, metrics in [
            ("step", self.step_metrics),
            ("episode", self.episode_metrics),
            ("update", self.metrics)
        ]:
            serializable_metrics[category] = {}
            for key, values in metrics.items():
                # Convert tensors to lists
                if isinstance(values, list):
                    serializable_values = []
                    for v in values:
                        if isinstance(v, torch.Tensor):
                            serializable_values.append(v.item())
                        else:
                            serializable_values.append(v)
                    serializable_metrics[category][key] = serializable_values
                else:
                    serializable_metrics[category][key] = values

        # Add metadata
        serializable_metrics["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "total_steps": self.step_count,
            "total_episodes": self.episode_count,
            "total_updates": self.update_count,
            "runtime_seconds": self.get_runtime()
        }

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

    def load_metrics(self, filepath: str):
        """
        Load metrics from file.

        Args:
            filepath: Path to load metrics from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Load metrics
        if "step" in data:
            self.step_metrics.update(data["step"])
        if "episode" in data:
            self.episode_metrics.update(data["episode"])
        if "update" in data:
            self.metrics.update(data["update"])

        # Load metadata
        if "metadata" in data:
            metadata = data["metadata"]
            self.step_count = metadata.get("total_steps", 0)
            self.episode_count = metadata.get("total_episodes", 0)
            self.update_count = metadata.get("total_updates", 0)

    def close(self):
        """Close all loggers and save final metrics."""
        # Save final metrics
        self.save_metrics()

        # Close TensorBoard
        if self.tb_writer:
            self.tb_writer.close()

        # Close wandb
        if self.use_wandb:
            try:
                wandb.finish()
            except:
                pass


class PerformanceProfiler:
    """
    Performance profiler for hierarchical RL training.
    """

    def __init__(self):
        self.timers = defaultdict(list)
        self.active_timers = {}

    def start_timer(self, name: str):
        """Start a timer."""
        self.active_timers[name] = time.time()

    def end_timer(self, name: str):
        """End a timer and record duration."""
        if name in self.active_timers:
            duration = time.time() - self.active_timers[name]
            self.timers[name].append(duration)
            del self.active_timers[name]
            return duration
        return 0.0

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics."""
        stats = {}
        for name, times in self.timers.items():
            if times:
                stats[name] = {
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "total": np.sum(times),
                    "count": len(times)
                }
        return stats

    def reset(self):
        """Reset all timers."""
        self.timers.clear()
        self.active_timers.clear()
