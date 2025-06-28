"""
Environment factory for creating RLGym environments.
Based on the working implementation from Rlbot-thesis.
Optimized for the Andromeda2 hierarchical RL architecture.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, List, Union, Optional, Any, Tuple
from rlgym_sim import make as rlgym_make
from rlgym_sim.envs import Match
from rlgym_sim.utils.gamestates import GameState
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.reward_functions.common_rewards import (
    VelocityPlayerToBallReward,
    VelocityBallToGoalReward,
    EventReward
)
from rlgym_sim.utils.reward_functions.combined_reward import CombinedReward
from rlgym_sim.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym_sim.utils.action_parsers.discrete_act import DiscreteAction
from rlgym_sim.utils.state_setters.default_state import DefaultState
from rlgym_sim.utils.state_setters.random_state import RandomState
from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM
import random


class RocketLeagueEnvFactory:
    """Factory for creating RocketLeague environments with different configurations."""

    @staticmethod
    def create_standard_env(
        num_players: int = 2,
        team_size: int = 1,
        obs_builder=None,
        action_parser=None,
        state_setter=None,
        reward_fn=None,
        terminal_conditions=None,
        spawn_opponents: bool = True,
        auto_detect_demos: bool = True,
        game_speed: int = 100,
        tick_skip: int = 8,
        render: bool = False,
        **kwargs
    ) -> Match:
        """
        Create a standard RocketLeague environment.

        Args:
            num_players: Total number of players
            team_size: Number of players per team
            obs_builder: Observation builder
            action_parser: Action parser
            state_setter: State setter
            reward_fn: Reward function
            terminal_conditions: Terminal conditions
            spawn_opponents: Whether to spawn opponents
            auto_detect_demos: Whether to auto-detect demos
            game_speed: Game speed multiplier
            tick_skip: Number of ticks to skip
            render: Whether to render

        Returns:
            Match environment
        """
        # Default components
        if obs_builder is None:
            obs_builder = AdvancedObs()

        if action_parser is None:
            action_parser = DiscreteAction()

        if state_setter is None:
            state_setter = DefaultState()

        if reward_fn is None:
            reward_fn = CombinedReward(
                (
                    VelocityPlayerToBallReward(),
                    VelocityBallToGoalReward(),
                    EventReward(
                        team_goal=100.0,
                        concede=-100.0,
                        shot=5.0,
                        save=30.0,
                        demo=10.0,
                    ),
                ),
                (0.1, 0.1, 1.0)
            )

        if terminal_conditions is None:
            terminal_conditions = [
                TimeoutCondition(225),  # 3.75 minutes
                GoalScoredCondition()
            ]

        # Create match object
        match = Match(
            reward_function=reward_fn,
            terminal_conditions=terminal_conditions,
            obs_builder=obs_builder,
            action_parser=action_parser,
            state_setter=state_setter,
            team_size=team_size,
            spawn_opponents=spawn_opponents
        )

        return match

    @staticmethod
    def create_1v1_env(**kwargs) -> Match:
        """Create a 1v1 environment."""
        return RocketLeagueEnvFactory.create_standard_env(
            num_players=2, team_size=1, **kwargs
        )

    @staticmethod
    def create_2v2_env(**kwargs) -> Match:
        """Create a 2v2 environment."""
        return RocketLeagueEnvFactory.create_standard_env(
            num_players=4, team_size=2, **kwargs
        )

    @staticmethod
    def create_3v3_env(**kwargs) -> Match:
        """Create a 3v3 environment."""
        return RocketLeagueEnvFactory.create_standard_env(
            num_players=6, team_size=3, **kwargs
        )

    @staticmethod
    def create_training_env(training_type: str = "basic", **kwargs) -> Match:
        """
        Create a training environment with specific configurations.

        Args:
            training_type: Type of training ('basic', 'aerial', 'dribbling', 'goalkeeping')
            **kwargs: Additional arguments

        Returns:
            Match environment configured for the specified training type
        """
        if training_type == "basic":
            return RocketLeagueEnvFactory.create_1v1_env(**kwargs)

        elif training_type == "aerial":
            # Custom state setter for aerial training
            state_setter = RandomState(
                ball_rand_speed=True,
                cars_rand_speed=True,
                cars_on_ground=False,
            )
            return RocketLeagueEnvFactory.create_1v1_env(
                state_setter=state_setter,
                **kwargs
            )

        elif training_type == "dribbling":
            # Custom reward for dribbling training
            reward_fn = CombinedReward(
                (
                    VelocityPlayerToBallReward(),
                    EventReward(touch=1.0),
                ),
                (1.0, 0.5)
            )
            return RocketLeagueEnvFactory.create_1v1_env(
                reward_fn=reward_fn,
                **kwargs
            )

        elif training_type == "goalkeeping":
            # Custom state setter for goalkeeping training
            state_setter = RandomState(
                ball_rand_speed=True,
                cars_rand_speed=False,
            )
            reward_fn = CombinedReward(
                (
                    EventReward(save=100.0, concede=-100.0),
                    VelocityPlayerToBallReward(),
                ),
                (1.0, 0.1)
            )
            return RocketLeagueEnvFactory.create_1v1_env(
                state_setter=state_setter,
                reward_fn=reward_fn,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown training type: {training_type}")


class HierarchicalRLEnvWrapper:
    """
    Wrapper for hierarchical RL environments.

    This wrapper adds hierarchical RL capabilities to standard environments,
    including goal vectors and intrinsic rewards.
    """

    def __init__(
        self,
        env: Union[Match, gym.Env],
        goal_vector_dim: int = 12,
        planner_update_freq: int = 8,
        intrinsic_reward_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the hierarchical RL wrapper.

        Args:
            env: Base environment (Match or gymnasium.Env)
            goal_vector_dim: Dimension of the goal vector
            planner_update_freq: How often the planner updates (in steps)
            intrinsic_reward_weights: Weights for intrinsic reward components
        """
        self.env = env
        self.is_match_env = isinstance(env, Match)
        self.goal_vector_dim = goal_vector_dim
        self.planner_update_freq = planner_update_freq
        self.step_count = 0
        self.current_goal_vector = np.zeros(goal_vector_dim)

        # Default intrinsic reward weights
        if intrinsic_reward_weights is None:
            self.intrinsic_reward_weights = {
                'car_velocity': 1.0,
                'ball_velocity': 1.0,
                'car_to_ball_pos': 1.0,
                'ball_to_goal_pos': 1.0
            }
        else:
            self.intrinsic_reward_weights = intrinsic_reward_weights

        # Store last state for intrinsic reward calculation
        self.last_state = None

    def reset(self, **kwargs):
        """Reset the environment."""
        if self.is_match_env:
            # For Match objects, we need to create a dummy state first
            # then let the state setter generate the proper initial state
            try:
                # Try to get the current state or create a dummy one
                if hasattr(self.env, '_prev_state') and self.env._prev_state is not None:
                    dummy_state = self.env._prev_state
                else:
                    # Create a minimal dummy state for initialization
                    from rlgym_sim.utils.gamestates import GameState
                    dummy_state = GameState()

                # Use the state setter to generate the initial state
                initial_state = self.env._state_setter.reset(dummy_state)
                state = self.env.episode_reset(initial_state)
                obs = self.env.build_observations(state)
            except Exception as e:
                # If that fails, try calling episode_reset without state_setter
                print(f"Warning: State setter failed ({e}), trying direct reset")
                try:
                    # Create a basic initial state using the state setter's build method
                    initial_state = self.env._state_setter.build_wrapper(team_size=self.env._team_size, spawn_opponents=self.env._spawn_opponents)
                    state = self.env.episode_reset(initial_state)
                    obs = self.env.build_observations(state)
                except Exception as e2:
                    print(f"Warning: Direct reset also failed ({e2}), using reset() method")
                    # Last resort: use gymnasium-style reset if available
                    obs = self.env.reset(**kwargs) if hasattr(self.env, 'reset') else None
                    if obs is None:
                        raise RuntimeError(f"Could not reset Match environment: {e2}")
        else:
            # For gymnasium environments
            obs = self.env.reset(**kwargs)

        self.step_count = 0
        self.current_goal_vector = np.zeros(self.goal_vector_dim)
        self.last_state = None
        return obs

    def step(self, action):
        """Step the environment with hierarchical RL modifications."""
        if self.is_match_env:
            # For Match objects
            state = self.env.step(action)
            obs = self.env.build_observations(state)
            reward = self.env.last_touch_reward
            done = any(self.env.last_touch_reward)
            info = {'state': state}
        else:
            # For gymnasium environments
            obs, reward, done, truncated, info = self.env.step(action)
            state = info.get('state', None)

        # Update step count
        self.step_count += 1

        # Update goal vector periodically
        if self.step_count % self.planner_update_freq == 0:
            self.current_goal_vector = self._update_goal_vector(state)

        # Calculate intrinsic rewards
        intrinsic_reward = self._calculate_intrinsic_rewards(state)

        # Combine extrinsic and intrinsic rewards
        if isinstance(reward, (list, np.ndarray)):
            total_reward = [r + intrinsic_reward for r in reward]
        else:
            total_reward = reward + intrinsic_reward

        # Add goal vector to info
        info['goal_vector'] = self.current_goal_vector
        info['intrinsic_reward'] = intrinsic_reward

        self.last_state = state

        return obs, total_reward, done, info

    def _update_goal_vector(self, state):
        """Update the goal vector based on current state."""
        # This is a simplified goal vector update
        # In practice, this would be learned by a higher-level policy
        goal_vector = np.random.randn(self.goal_vector_dim) * 0.1
        return goal_vector

    def _calculate_intrinsic_rewards(self, state):
        """Calculate intrinsic rewards based on state transitions."""
        if self.last_state is None or state is None:
            return 0.0

        intrinsic_reward = 0.0

        if self.is_match_env and hasattr(state, 'players') and hasattr(state, 'ball'):
            # Calculate various intrinsic reward components
            try:
                # Car velocity reward
                car_vel = np.linalg.norm(state.players[0].car_data.linear_velocity)
                intrinsic_reward += self.intrinsic_reward_weights.get('car_velocity', 0.0) * car_vel * 0.01

                # Ball velocity reward
                ball_vel = np.linalg.norm(state.ball.linear_velocity)
                intrinsic_reward += self.intrinsic_reward_weights.get('ball_velocity', 0.0) * ball_vel * 0.01

                # Car to ball distance reward (negative reward for being far)
                car_pos = np.array(state.players[0].car_data.position)
                ball_pos = np.array(state.ball.position)
                car_to_ball_dist = np.linalg.norm(car_pos - ball_pos)
                intrinsic_reward -= self.intrinsic_reward_weights.get('car_to_ball_pos', 0.0) * car_to_ball_dist * 0.001

                # Ball to goal distance reward
                goal_pos = np.array([0, 5120, 0])  # Orange goal position
                ball_to_goal_dist = np.linalg.norm(ball_pos - goal_pos)
                intrinsic_reward -= self.intrinsic_reward_weights.get('ball_to_goal_pos', 0.0) * ball_to_goal_dist * 0.0001

            except (AttributeError, IndexError):
                # Handle cases where state structure is different
                pass

        return intrinsic_reward

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self.env, name)


def make_env(
    env_type: str = "1v1",
    hierarchical: bool = False,
    use_rlgym_make: bool = False,
    **kwargs
) -> Union[Match, HierarchicalRLEnvWrapper]:
    """
    Factory function to create environments.

    Args:
        env_type: Type of environment ('1v1', '2v2', '3v3', 'training')
        hierarchical: Whether to wrap with hierarchical RL wrapper
        use_rlgym_make: Whether to use rlgym_sim.make() for full gym interface
                       (requires collision mesh files in ./collision_meshes/)
        **kwargs: Additional arguments passed to environment creation

    Returns:
        Environment instance. Returns Match object by default, or Gym-wrapped
        environment if use_rlgym_make=True and collision meshes are available.

    Note:
        - Match objects have methods like episode_reset(), get_result(), etc.
        - For standard gym interface (reset(), step()), use use_rlgym_make=True
        - Collision meshes are required for the full simulator to work
    """
    if env_type == "1v1":
        match = RocketLeagueEnvFactory.create_1v1_env(**kwargs)
    elif env_type == "2v2":
        match = RocketLeagueEnvFactory.create_2v2_env(**kwargs)
    elif env_type == "3v3":
        match = RocketLeagueEnvFactory.create_3v3_env(**kwargs)
    elif env_type == "training":
        match = RocketLeagueEnvFactory.create_training_env(**kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    # Optionally wrap with rlgym_make for full gym interface
    if use_rlgym_make:
        try:
            env = rlgym_make(
                match=match,
                tick_skip=kwargs.get('tick_skip', 8),
                **{k: v for k, v in kwargs.items() if k not in ['tick_skip']}
            )
        except Exception as e:
            print(f"Warning: Could not create gym environment: {e}")
            print("Falling back to Match object. You may need collision mesh files.")
            env = match
    else:
        env = match

    if hierarchical:
        # Filter kwargs for HierarchicalRLEnvWrapper - only pass valid arguments
        hierarchical_kwargs = {}
        valid_hierarchical_args = ['goal_vector_dim', 'planner_update_freq', 'intrinsic_reward_weights']
        for key in valid_hierarchical_args:
            if key in kwargs:
                hierarchical_kwargs[key] = kwargs[key]

        env = HierarchicalRLEnvWrapper(env, **hierarchical_kwargs)

    return env


def setup_instructions():
    """
    Print setup instructions for the environment.
    """
    print("""
RLGym Environment Setup Instructions:

1. Install rlgym_sim:
   pip install rlgym-sim

2. For full simulator support (optional):
   - Download collision mesh files
   - Place them in ./collision_meshes/ directory
   - Use use_rlgym_make=True when creating environments

3. Basic usage:
   # Create a basic 1v1 environment
   env = make_env(env_type="1v1")

   # Use Match API:
   state = env.episode_reset()
   actions = {...}  # Your actions
   state = env.get_result(actions)

   # Or use gym-like interface (requires collision meshes):
   env = make_env(env_type="1v1", use_rlgym_make=True)
   obs = env.reset()
   obs, reward, done, info = env.step(action)

4. Hierarchical RL:
   env = make_env(env_type="1v1", hierarchical=True)

5. Training environments:
   env = make_env(env_type="training", training_type="aerial")
""")


if __name__ == "__main__":
    setup_instructions()
