"""
Vectorized environment wrapper for parallel training.
Based on the working implementation from Rlbot-thesis.
Optimized for the Andromeda2 hierarchical RL architecture.
"""

import numpy as np
import torch
import multiprocessing as mp
import gymnasium as gym
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from collections import deque
import pickle
import traceback
from .factory import make_env, HierarchicalRLEnvWrapper


def _worker_process(
    conn: mp.connection.Connection,
    env_indices: List[int],
    env_config: Dict[str, Any]
):
    """Worker process function."""
    try:
        # Create environments for this worker
        envs = []
        for _ in env_indices:
            env = make_env(**env_config)
            envs.append(env)

        while True:
            try:
                cmd, data = conn.recv()

                if cmd == 'reset':
                    observations = []
                    for env in envs:
                        if hasattr(env, 'reset'):
                            obs = env.reset()
                        else:
                            # For Match objects, use episode_reset with state setter
                            try:
                                from rlgym_sim.utils.gamestates import GameState
                                dummy_state = GameState()
                                # Try to build a proper initial state
                                if hasattr(env, '_state_setter'):
                                    try:
                                        initial_state = env._state_setter.build_wrapper(
                                            team_size=getattr(env, 'team_size', 1),
                                            spawn_opponents=getattr(env, '_spawn_opponents', True)
                                        )
                                    except:
                                        # Fallback: use DefaultState to create initial state
                                        from rlgym_sim.utils.state_setters.default_state import DefaultState
                                        state_setter = DefaultState()
                                        initial_state = state_setter.build_wrapper(team_size=1, spawn_opponents=True)

                                    state = env.episode_reset(initial_state)
                                    obs = env.build_observations(state)
                                else:
                                    # Last resort: create minimal state
                                    from rlgym_sim.utils.state_setters.default_state import DefaultState
                                    state_setter = DefaultState()
                                    initial_state = state_setter.build_wrapper(team_size=1, spawn_opponents=True)
                                    state = env.episode_reset(initial_state)
                                    obs = env.build_observations(state)
                            except Exception as e:
                                print(f"Error resetting Match environment: {e}")
                                # Return empty observation as fallback
                                obs = []
                        observations.append(obs)
                    conn.send(('reset_result', observations))

                elif cmd == 'step':
                    actions = data
                    results = []
                    for i, (env, action) in enumerate(zip(envs, actions)):
                        try:
                            if hasattr(env, 'step'):
                                if isinstance(env, HierarchicalRLEnvWrapper) or hasattr(env, 'is_match_env'):
                                    # For our wrapped environments
                                    obs, reward, done, info = env.step(action)
                                else:
                                    # For standard gym environments
                                    step_result = env.step(action)
                                    if len(step_result) == 5:  # New gym API
                                        obs, reward, terminated, truncated, info = step_result
                                        done = terminated or truncated
                                    else:  # Old gym API
                                        obs, reward, done, info = step_result
                                results.append((obs, reward, done, info))
                            else:
                                # For Match objects
                                state = env.get_result(action)
                                obs = env.build_observations(state)
                                reward = env.get_rewards()
                                done = env.is_done()
                                info = {'state': state}
                                results.append((obs, reward, done, info))
                        except Exception as e:
                            print(f"Error in env {env_indices[i]}: {e}")
                            # Reset environment on error
                            obs = env.reset()
                            results.append((obs, 0.0, True, {'error': str(e)}))
                    conn.send(('step_result', results))

                elif cmd == 'get_specs':
                    # Get specs from first environment
                    if envs:
                        # Handle reset for specs
                        if hasattr(envs[0], 'reset'):
                            dummy_obs = envs[0].reset()
                        else:
                            # For Match objects
                            try:
                                from rlgym_sim.utils.state_setters.default_state import DefaultState
                                state_setter = DefaultState()
                                initial_state = state_setter.build_wrapper(team_size=1, spawn_opponents=True)
                                state = envs[0].episode_reset(initial_state)
                                dummy_obs = envs[0].build_observations(state)
                            except Exception as e:
                                print(f"Error getting specs from Match environment: {e}")
                                dummy_obs = []

                        if isinstance(dummy_obs, list):
                            obs_space = len(dummy_obs[0]) if dummy_obs else 0
                            num_agents = len(dummy_obs)
                        else:
                            obs_space = len(dummy_obs) if hasattr(dummy_obs, '__len__') else 0
                            num_agents = 1

                        # Try to get action space
                        if hasattr(envs[0], 'action_space'):
                            action_space = envs[0].action_space
                        elif hasattr(envs[0], '_action_parser'):
                            action_space = envs[0]._action_parser.get_action_space()
                        else:
                            action_space = None

                        specs = {
                            'observation_space': obs_space,
                            'action_space': action_space,
                            'num_agents': num_agents
                        }
                    else:
                        specs = {'observation_space': 0, 'action_space': None, 'num_agents': 1}
                    conn.send(('specs_result', specs))

                elif cmd == 'close':
                    for env in envs:
                        if hasattr(env, 'close'):
                            env.close()
                    break

                elif cmd == 'render':
                    render_results = []
                    for env in envs:
                        if hasattr(env, 'render'):
                            render_result = env.render()
                            render_results.append(render_result)
                        else:
                            render_results.append(None)
                    conn.send(('render_result', render_results))

                else:
                    conn.send(('error', f'Unknown command: {cmd}'))

            except EOFError:
                break
            except Exception as e:
                conn.send(('error', str(e)))
                traceback.print_exc()

    except Exception as e:
        print(f"Worker process error: {e}")
        traceback.print_exc()
    finally:
        conn.close()


class VectorizedRLGymEnv:
    """
    Vectorized wrapper for RLGym environments that supports parallel execution.
    Based on the working implementation from Rlbot-thesis.
    """

    def __init__(
        self,
        num_envs: int,
        env_config: Dict[str, Any],
        num_workers: Optional[int] = None,
        timeout: float = 60.0,
        shared_memory: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments
            env_config: Configuration for each environment
            num_workers: Number of worker processes (defaults to num_envs)
            timeout: Timeout for operations in seconds
            shared_memory: Whether to use shared memory for observations
            device: Device for tensor operations
        """
        self.num_envs = num_envs
        self.env_config = env_config
        self.timeout = timeout
        self.device = device
        self.shared_memory = shared_memory

        if num_workers is None:
            num_workers = min(num_envs, mp.cpu_count())
        self.num_workers = num_workers

        # Initialize worker processes
        self._init_workers()

        # Get environment specs
        self._get_env_specs()

        # Initialize state tracking
        self.reset_async_called = False
        self.step_async_called = False

    def _init_workers(self):
        """Initialize worker processes."""
        self.workers = []
        self.parent_conns = []
        self.processes = []

        # Create worker processes
        for i in range(self.num_workers):
            parent_conn, child_conn = mp.Pipe()

            # Determine which environments this worker handles
            envs_per_worker = self.num_envs // self.num_workers
            extra_envs = self.num_envs % self.num_workers

            start_idx = i * envs_per_worker + min(i, extra_envs)
            end_idx = start_idx + envs_per_worker + (1 if i < extra_envs else 0)

            worker_env_indices = list(range(start_idx, end_idx))

            # Create a clean copy of env_config without unpicklable objects
            clean_env_config = {}
            for key, value in self.env_config.items():
                try:
                    # Test if value is picklable
                    pickle.dumps(value)
                    clean_env_config[key] = value
                except (TypeError, AttributeError):
                    # Skip unpicklable objects
                    print(f"Warning: Skipping unpicklable config key: {key}")
                    continue

            process = mp.Process(
                target=_worker_process,
                args=(
                    child_conn,
                    worker_env_indices,
                    clean_env_config
                )
            )
            process.daemon = True
            process.start()

            self.workers.append((parent_conn, worker_env_indices))
            self.parent_conns.append(parent_conn)
            self.processes.append(process)

    def _get_env_specs(self):
        """Get environment specifications."""
        # Send get_specs command to first worker
        self.parent_conns[0].send(('get_specs', None))

        try:
            cmd, specs = self.parent_conns[0].recv()
            if cmd == 'specs_result':
                self.observation_space = specs['observation_space']
                self.action_space = specs['action_space']
                self.num_agents = specs['num_agents']
            else:
                raise RuntimeError(f"Failed to get environment specs: {specs}")
        except Exception as e:
            raise RuntimeError(f"Failed to get environment specs: {e}")

    def reset(self, **kwargs):
        """Reset all environments."""
        # Send reset command to all workers
        for conn in self.parent_conns:
            conn.send(('reset', None))

        # Collect results
        all_observations = []
        for conn in self.parent_conns:
            cmd, observations = conn.recv()
            if cmd == 'reset_result':
                all_observations.extend(observations)
            else:
                raise RuntimeError(f"Reset failed: {observations}")

        # Format observations
        return self._format_observations(all_observations)

    def step(self, actions):
        """Step all environments with given actions."""
        if not isinstance(actions, (list, np.ndarray)):
            actions = [actions] * self.num_envs

        # Distribute actions to workers
        action_idx = 0
        for (conn, worker_env_indices) in self.workers:
            worker_actions = actions[action_idx:action_idx + len(worker_env_indices)]
            conn.send(('step', worker_actions))
            action_idx += len(worker_env_indices)

        # Collect results
        all_results = []
        for (conn, _) in self.workers:
            cmd, results = conn.recv()
            if cmd == 'step_result':
                all_results.extend(results)
            else:
                raise RuntimeError(f"Step failed: {results}")

        # Separate observations, rewards, dones, infos
        observations, rewards, dones, infos = zip(*all_results)

        # Format outputs
        observations = self._format_observations(list(observations))
        rewards = np.array(rewards)
        dones = np.array(dones)
        infos = list(infos)

        return observations, rewards, dones, infos

    def _format_observations(self, observations: List[Any]) -> np.ndarray:
        """Format observations into a consistent format."""
        if not observations:
            return np.array([])

        # Handle different observation types
        formatted_obs = []
        for obs in observations:
            if isinstance(obs, (list, tuple)):
                # Multi-agent observations
                if len(obs) > 0 and isinstance(obs[0], (list, np.ndarray)):
                    # Flatten multi-agent observations
                    flat_obs = []
                    for agent_obs in obs:
                        if isinstance(agent_obs, np.ndarray):
                            flat_obs.extend(agent_obs.flatten())
                        else:
                            flat_obs.extend(np.array(agent_obs).flatten())
                    formatted_obs.append(np.array(flat_obs))
                else:
                    # Single agent observation in list format
                    formatted_obs.append(np.array(obs).flatten())
            elif isinstance(obs, np.ndarray):
                formatted_obs.append(obs.flatten())
            elif isinstance(obs, dict):
                # Dictionary observations - flatten all values
                flat_obs = []
                for key in sorted(obs.keys()):
                    val = obs[key]
                    if isinstance(val, np.ndarray):
                        flat_obs.extend(val.flatten())
                    elif isinstance(val, (list, tuple)):
                        flat_obs.extend(np.array(val).flatten())
                    else:
                        flat_obs.append(float(val))
                formatted_obs.append(np.array(flat_obs))
            else:
                # Scalar or unknown type
                formatted_obs.append(np.array([float(obs)]))

        # Ensure all observations have the same shape
        if formatted_obs:
            max_len = max(len(obs) for obs in formatted_obs)
            padded_obs = []
            for obs in formatted_obs:
                if len(obs) < max_len:
                    padded = np.zeros(max_len)
                    padded[:len(obs)] = obs
                    padded_obs.append(padded)
                else:
                    padded_obs.append(obs[:max_len])
            return np.array(padded_obs)
        else:
            return np.array([])

    def render(self, mode: str = 'human', **kwargs):
        """Render the environments."""
        # Send render command to all workers
        for conn in self.parent_conns:
            conn.send(('render', None))

        # Collect results
        all_renders = []
        for conn in self.parent_conns:
            cmd, renders = conn.recv()
            if cmd == 'render_result':
                all_renders.extend(renders)

        return all_renders

    def close(self):
        """Close all environments and workers."""
        # Send close command to all workers
        for conn in self.parent_conns:
            try:
                conn.send(('close', None))
            except (BrokenPipeError, ConnectionResetError):
                pass

        # Wait for processes to finish and clean up
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()

        # Close connections
        for conn in self.parent_conns:
            try:
                conn.close()
            except:
                pass

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


class AsyncVectorizedEnv:
    """
    Asynchronous vectorized environment wrapper.

    This version allows for non-blocking step operations,
    which can improve performance when environments have
    variable step times.
    """

    def __init__(
        self,
        num_envs: int,
        env_config: Dict[str, Any],
        max_workers: Optional[int] = None,
        timeout: float = 60.0,
        **kwargs
    ):
        """
        Initialize asynchronous vectorized environment.

        Args:
            num_envs: Number of parallel environments
            env_config: Configuration for each environment
            max_workers: Maximum number of worker threads
            timeout: Timeout for operations in seconds
            **kwargs: Additional arguments
        """
        self.num_envs = num_envs
        self.env_config = env_config
        self.timeout = timeout

        if max_workers is None:
            max_workers = min(num_envs, mp.cpu_count())

        # Create thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Create individual environments
        self.envs = []
        for i in range(num_envs):
            env = make_env(**env_config)
            self.envs.append(env)

        # State tracking
        self.reset_futures = None
        self.step_futures = None

    def reset_async(self):
        """Start asynchronous reset of all environments."""
        self.reset_futures = []
        for env in self.envs:
            future = self.executor.submit(env.reset)
            self.reset_futures.append(future)

    def reset_wait(self, timeout: Optional[float] = None):
        """Wait for asynchronous resets to complete."""
        if self.reset_futures is None:
            raise RuntimeError("reset_async() must be called first")

        timeout = timeout or self.timeout
        observations = []

        for future in self.reset_futures:
            try:
                obs = future.result(timeout=timeout)
                observations.append(obs)
            except Exception as e:
                raise RuntimeError(f"Reset failed: {e}")

        self.reset_futures = None
        return observations

    def reset(self, **kwargs):
        """Reset all environments synchronously."""
        self.reset_async()
        return self.reset_wait()

    def step_async(self, actions):
        """Start asynchronous step of all environments."""
        if not isinstance(actions, (list, np.ndarray)):
            actions = [actions] * self.num_envs

        self.step_futures = []
        for env, action in zip(self.envs, actions):
            future = self.executor.submit(env.step, action)
            self.step_futures.append(future)

    def step_wait(self, timeout: Optional[float] = None):
        """Wait for asynchronous steps to complete."""
        if self.step_futures is None:
            raise RuntimeError("step_async() must be called first")

        timeout = timeout or self.timeout
        results = []

        for future in self.step_futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                raise RuntimeError(f"Step failed: {e}")

        self.step_futures = None

        # Separate observations, rewards, dones, infos
        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)

    def step(self, actions):
        """Step all environments synchronously."""
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        """Close all environments and executor."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
        self.executor.shutdown(wait=True)

    def __getattr__(self, name):
        """Delegate attribute access to the first environment."""
        return getattr(self.envs[0], name)


def make_vectorized_env(
    num_envs: int,
    env_config: Dict[str, Any],
    async_env: bool = False,
    **kwargs
) -> Union[VectorizedRLGymEnv, AsyncVectorizedEnv]:
    """
    Factory function to create vectorized environments.

    Args:
        num_envs: Number of parallel environments
        env_config: Configuration for each environment
        async_env: Whether to use asynchronous version
        **kwargs: Additional arguments

    Returns:
        Vectorized environment instance
    """
    if async_env:
        return AsyncVectorizedEnv(num_envs=num_envs, env_config=env_config, **kwargs)
    else:
        return VectorizedRLGymEnv(num_envs=num_envs, env_config=env_config, **kwargs)


def batch_observations(observations: List[np.ndarray]) -> np.ndarray:
    """
    Batch a list of observations into a single array.

    Args:
        observations: List of observation arrays

    Returns:
        Batched observations array
    """
    if not observations:
        return np.array([])

    # Ensure all observations have the same shape
    max_shape = tuple(max(obs.shape[i] for obs in observations)
                     for i in range(len(observations[0].shape)))

    batched = np.zeros((len(observations),) + max_shape, dtype=observations[0].dtype)

    for i, obs in enumerate(observations):
        # Use slicing to handle different shapes
        slices = tuple(slice(0, s) for s in obs.shape)
        batched[i][slices] = obs

    return batched


def unbatch_actions(actions: np.ndarray, num_envs: int) -> List[np.ndarray]:
    """
    Unbatch a single action array into a list of individual actions.

    Args:
        actions: Batched actions array
        num_envs: Number of environments

    Returns:
        List of individual action arrays
    """
    if actions.ndim == 1:
        # Single action for all environments
        return [actions] * num_envs
    else:
        # Different actions for each environment
        return [actions[i] for i in range(min(len(actions), num_envs))]


class VectorizedEnvPool:
    """
    Environment pool for managing multiple vectorized environments.

    This can be useful for very large scale training where you want
    to manage multiple groups of vectorized environments.
    """

    def __init__(
        self,
        num_pools: int,
        envs_per_pool: int,
        env_config: Dict[str, Any],
        **kwargs
    ):
        """
        Initialize environment pool.

        Args:
            num_pools: Number of vectorized environment pools
            envs_per_pool: Number of environments per pool
            env_config: Configuration for each environment
            **kwargs: Additional arguments for vectorized environments
        """
        self.num_pools = num_pools
        self.envs_per_pool = envs_per_pool
        self.total_envs = num_pools * envs_per_pool

        # Create vectorized environment pools
        self.env_pools = []
        for i in range(num_pools):
            pool = VectorizedRLGymEnv(
                num_envs=envs_per_pool,
                env_config=env_config,
                **kwargs
            )
            self.env_pools.append(pool)

    def reset(self, pool_idx: Optional[int] = None):
        """Reset environments in specified pool or all pools."""
        if pool_idx is not None:
            return self.env_pools[pool_idx].reset()
        else:
            observations = []
            for pool in self.env_pools:
                obs = pool.reset()
                observations.extend(obs)
            return observations

    def step(self, actions, pool_idx: Optional[int] = None):
        """Step environments in specified pool or all pools."""
        if pool_idx is not None:
            return self.env_pools[pool_idx].step(actions)
        else:
            # Distribute actions across pools
            all_observations = []
            all_rewards = []
            all_dones = []
            all_infos = []

            action_idx = 0
            for pool in self.env_pools:
                pool_actions = actions[action_idx:action_idx + self.envs_per_pool]
                obs, rewards, dones, infos = pool.step(pool_actions)

                all_observations.extend(obs)
                all_rewards.extend(rewards)
                all_dones.extend(dones)
                all_infos.extend(infos)

                action_idx += self.envs_per_pool

            return all_observations, all_rewards, all_dones, all_infos

    def close(self):
        """Close all environment pools."""
        for pool in self.env_pools:
            pool.close()
