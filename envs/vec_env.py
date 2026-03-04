"""
vec_env.py
===========

This module implements vectorized wrappers for the inventory management environments.

UPDATED VERSION:
This version implements TRUE PARALLEL execution using `multiprocessing`.
It replaces the previous sequential implementation to maximize CPU utilization
during training.

Classes:
    MultiDiscrete: Helper for multi-discrete action spaces.
    SubprocVecEnv: True parallel vectorized environment using multiprocessing.
    DummyVecEnv: Single-thread wrapper for evaluation/debugging.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Any, Callable


# ---------------------------------------------------------------------------
# Lightweight space helpers (replaces gym.spaces to avoid heavy dependency)
# ---------------------------------------------------------------------------

class _Space:
    """Minimal base class mirroring gym._Space."""
    def __init__(self, shape=None, dtype=None):
        self._shape = shape
        self.dtype = dtype
    @property
    def shape(self):
        return self._shape

class Discrete(_Space):
    """Discrete action space with *n* actions {0, 1, …, n-1}."""
    def __init__(self, n: int):
        super().__init__(shape=(), dtype=np.int64)
        self.n = n
    def sample(self):
        return int(np.random.randint(self.n))
    def __repr__(self):
        return f"Discrete({self.n})"

class Box(_Space):
    """Continuous box observation space."""
    def __init__(self, low, high, shape=None, dtype=np.float32):
        super().__init__(shape=shape, dtype=dtype)
        self.low = low
        self.high = high

# --- Helper Functions for Multiprocessing ---

def worker(remote, parent_remote, env_fn_wrapper):
    """
    Worker process that holds a single environment instance.
    Waits for commands from the main process via pipe.
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                actions, one_hot = data
                ob, reward, done, info = env.step(actions, one_hot=one_hot)
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                train_mode = data
                ob = env.reset(train=train_mode)
                remote.send(ob)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'call_method':
                method_name, args = data
                method = getattr(env, method_name)
                remote.send(method(*args) if args else method())
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

# --- Environment Classes ---

class MultiDiscrete(_Space):
    """A simple multi-discrete action space."""
    def __init__(self, array_of_param_array: List[Tuple[int, int]]):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high - self.low + 1)
        super().__init__(shape=(self.num_discrete_space,), dtype=np.int64)

    def sample(self) -> List[int]:
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor((self.high - self.low + 1.) * random_array + self.low)]

    def contains(self, x: Any) -> bool:
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.num_discrete_space,)

    def __repr__(self) -> str:
        return f"MultiDiscrete{self.num_discrete_space}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MultiDiscrete) and np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


class SubprocVecEnv:
    """
    True parallel vectorized environment.
    Runs 'n_envs' on separate processes using multiprocessing Pipes.
    """
    def __init__(self, env_fns: List[Callable], n_envs: int = None):
        """
        env_fns: List of callables that create environments.
        """
        # Allow passing a single list of functions (compatible with standard VecEnv)
        # or a single function + n_envs (compatible with your old code style if adjusted)
        
        if isinstance(env_fns, list):
            self.env_fns = env_fns
        else:
            # Fallback for old style: env_fn, n_envs passed
            self.env_fns = [env_fns for _ in range(n_envs)]

        self.num_envs = len(self.env_fns)
        self.waiting = False
        self.closed = False

        # Use 'spawn' context for better compatibility (Windows/CUDA)
        ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        
        self.ps = [
            ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, self.env_fns)
        ]

        for p in self.ps:
            p.daemon = True  # Clean up if main process dies
            p.start()
        
        for remote in self.work_remotes:
            remote.close()

        # Get properties from the first environment to setup spaces
        self.remotes[0].send(('get_attr', 'agent_num'))
        self.num_agent = self.remotes[0].recv()
        
        self.remotes[0].send(('get_attr', 'obs_dim'))
        self.signal_obs_dim = self.remotes[0].recv()
        
        self.remotes[0].send(('get_attr', 'action_dim'))
        self.signal_action_dim = self.remotes[0].recv()

        # Build spaces similar to original code
        self.action_space: List[_Space] = []
        self.observation_space: List[_Space] = []
        share_obs_dim = 0
        
        for _ in range(self.num_agent):
            self.action_space.append(Discrete(self.signal_action_dim))
            self.observation_space.append(
                Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32)
            )
            share_obs_dim += self.signal_obs_dim
            
        self.share_observation_space = [
            Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def step(self, actions_batch: List[List[int]], one_hot: bool = False):
        self._assert_not_closed()
        
        # Send actions to all workers
        for remote, action in zip(self.remotes, actions_batch):
            remote.send(('step', (action, one_hot)))
            
        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        obs, rewards, dones, infos = zip(*results)
        
        # Original logic: Flatten reward list to plain numbers if they are lists
        # Handling this here in main process
        flat_rewards = []
        for r_agent in rewards:
            # r_agent is expected to be a list of rewards for each agent in that env
            flat_r = [r[0] if isinstance(r, list) else r for r in r_agent]
            flat_rewards.append(flat_r)
            
        return np.stack(obs), np.array(flat_rewards), np.stack(dones), infos

    def reset(self, train: bool = True):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', train))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def get_eval_bw_res(self) -> List[float]:
        # Get from first env
        self.remotes[0].send(('call_method', ('get_eval_bw_res', None)))
        return self.remotes[0].recv()

    def get_eval_num(self) -> int:
        self.remotes[0].send(('call_method', ('get_eval_num', None)))
        return self.remotes[0].recv()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a closed SubprocVecEnv"


class DummyVecEnv:
    """
    A dummy vectorised environment wrapping a single environment.
    Kept for evaluation purposes where parallelism is not strictly needed.
    """
    def __init__(self, env_fn):
        self.env = env_fn()
        self.num_envs = 1
        self.num_agent = self.env.agent_num
        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim
        
        self.observation_space = [
            Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]
        self.action_space = [Discrete(self.signal_action_dim) for _ in range(self.num_agent)]
        
        share_obs_dim = self.signal_obs_dim * self.num_agent
        self.share_observation_space = [
            Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def reset(self, train: bool = False):
        obs = self.env.reset(train=train)
        return np.array([obs])

    def step(self, actions_batch: List[List[int]], one_hot: bool = False):
        # actions_batch is expected to be a list of length 1
        next_obs, rewards, done, info = self.env.step(actions_batch[0], one_hot=one_hot)
        flat_rew = [r[0] if isinstance(r, list) else r for r in rewards]
        return np.array([next_obs]), np.array([flat_rew]), np.array([done]), [info]

    def get_eval_bw_res(self) -> List[float]:
        return self.env.get_eval_bw_res() if hasattr(self.env, "get_eval_bw_res") else []

    def get_eval_num(self) -> int:
        return self.env.get_eval_num() if hasattr(self.env, "get_eval_num") else 0