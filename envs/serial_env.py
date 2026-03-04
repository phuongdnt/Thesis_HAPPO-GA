"""
serial_env.py
==============

This module implements a simplified multi-echelon inventory management
environment for reinforcement learning.
FIXED VERSION:
- Added close() method to prevent multiprocessing crashes.
- Inherits from BaseInventoryEnv.
- Returns 'demand' and 'sales' in info for Pareto analysis.
"""

from __future__ import annotations

import numpy as np
import random
from typing import List, Tuple, Optional, Any
from pathlib import Path

# --- FIX: Import Base Class ---
from envs.base_env import BaseInventoryEnv
from data_loader import load_eval_data, generate_training_demand


class SerialInventoryEnv(BaseInventoryEnv): # --- FIX: Inheritance ---
    """Multi-echelon serial supply chain environment.

    Parameters
    ----------
    level_num : int
        Number of agents/echelons in the supply chain.
    lead_time : int
        Shipping lead time between consecutive levels.
    episode_len : int
        Length of each episode (number of time steps).
    action_dim : int
        Size of the discrete action space.
    init_inventory : int
        Initial inventory level for each agent.
    init_outstanding : int
        Initial outstanding (pipeline) orders.
    holding_cost : List[float]
        Per-unit holding cost for each agent.
    backlog_cost : List[float]
        Per-unit backorder cost for each agent.
    fixed_cost : float
        Fixed ordering cost.
    price_discount : bool
        If true, apply a price discount schedule.
    discount_schedule : List[float]
        Discount multipliers for buckets of order quantities.
    eval_data_dir : Optional[str]
        Path to a directory containing evaluation demand sequences.
    rng_seed : Optional[int]
        Seed for random number generation.
    """

    def __init__(
        self,
        level_num: int = 3,
        lead_time: int = 2,
        episode_len: int = 200,
        action_dim: int = 21,
        init_inventory: int = 10,
        init_outstanding: int = 10,
        holding_cost: Optional[List[float]] = None,
        backlog_cost: Optional[List[float]] = None,
        fixed_cost: float = 0.0,
        price_discount: bool = False,
        discount_schedule: Optional[List[float]] = None,
        eval_data_dir: Optional[str] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.level_num = level_num
        self.lead_time = lead_time
        self.episode_len = episode_len
        self.action_dim = action_dim
        self.init_inventory = init_inventory
        self.init_outstanding = init_outstanding
        self.holding_cost = holding_cost if holding_cost is not None else [1.0] * level_num
        self.backlog_cost = backlog_cost if backlog_cost is not None else [1.0] * level_num
        self.fixed_cost = fixed_cost
        self.price_discount = price_discount
        if discount_schedule is None:
            # Default discount schedule: no discount for any bucket.
            self.discount_schedule = [1.0 for _ in range((action_dim + 4) // 5)]
        else:
            self.discount_schedule = discount_schedule
        self.eval_data_dir = eval_data_dir
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        # Derived attributes
        self.agent_num = self.level_num
        self.obs_dim = self.lead_time + 3

        # Evaluation data
        if self.eval_data_dir:
            self.n_eval, self.eval_data = load_eval_data(self.eval_data_dir)
        else:
            self.n_eval, self.eval_data = 0, []
        self.eval_index: int = 0

        # Environment state (initialised in reset)
        self.inventory: List[int] = []
        self.backlog: List[int] = []
        self.backlog_history: List[List[int]] = []
        self.pipeline_orders: List[List[int]] = []
        self.demand_list: List[int] = []
        self.step_num: int = 0
        self.train: bool = True
        self.normalize: bool = True
        self.action_history: List[List[int]] = []
        self.record_act_sta: List[List[float]] = [[] for _ in range(self.level_num)]
        self.eval_bw_res: List[float] = []
        
        # NEW: Track demand and fulfilled for service level calculation
        self.demand_history: List[List[int]] = []
        self.fulfilled_history: List[List[int]] = []
        
        # For reward smoothing
        self.alpha: float = 0.5

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def reset(self, train: bool = True, normalize: bool = True) -> List[np.ndarray]:
        """Reset the environment for a new episode."""
        self.train = train
        self.normalize = normalize
        self.step_num = 0
        self.inventory = [self.init_inventory for _ in range(self.level_num)]
        self.backlog = [0 for _ in range(self.level_num)]
        self.backlog_history = [[] for _ in range(self.level_num)]
        self.pipeline_orders = [
            [self.init_outstanding for _ in range(self.lead_time)]
            for _ in range(self.level_num)
        ]
        self.action_history = [[] for _ in range(self.level_num)]
        
        # NEW: Reset demand and fulfilled history
        self.demand_history = [[] for _ in range(self.level_num)]
        self.fulfilled_history = [[] for _ in range(self.level_num)]
        
        # NEW: Track inventory levels over time
        self.inventory_history = [[] for _ in range(self.level_num)]
        
        # Load or generate demand sequence
        if not train:
            if not self.eval_data:
                raise RuntimeError(
                    "Evaluation data directory not provided or empty."
                )
            self.demand_list = self.eval_data[self.eval_index]
            self.eval_index = (self.eval_index + 1) % max(1, self.n_eval)
        else:
            # Generate random demand using the maximum possible action as the upper bound
            train_demand_max = 20 
            self.demand_list = generate_training_demand(
                self.episode_len, train_demand_max, distribution="uniform", seed=self.rng_seed
            )
        # Reset metrics for bullwhip measurement in evaluation
        self.record_act_sta = [[] for _ in range(self.level_num)]
        self.eval_bw_res = []
        # Initial observation
        return self._get_reset_obs()

    def step(self, actions: List[int], one_hot: bool = True) -> Tuple[List[np.ndarray], List[List[float]], List[bool], List[Any]]:
        """Advance the environment by one time step."""
        # Convert one-hot actions to integer indices
        if one_hot:
            act_idxs = [int(np.argmax(a)) for a in actions]
        else:
            act_idxs = [int(a) for a in actions]
        # Map indices to actual order quantities (identity mapping here)
        order_quantities = self._action_map(act_idxs)
        # Update state and compute raw rewards
        reward = self._state_update(order_quantities)
        # Get observations for next step
        next_obs = self._get_step_obs(order_quantities)
        # Process rewards (smooth across agents if training)
        processed_rewards = self._get_processed_rewards(reward)
        # Determine done flag
        done_flag = self.step_num >= self.episode_len
        done = [done_flag for _ in range(self.agent_num)]
        
        # --- FIX: RETURN INFO WITH DEMAND & SALES ---
        info = []
        for i in range(self.agent_num):
            info.append({
                'demand': self.demand_history[i][-1], 
                'sales': self.fulfilled_history[i][-1]
            })
        # --------------------------------------------
        
        return next_obs, processed_rewards, done, info

    def get_eval_num(self) -> int:
        return self.n_eval

    def get_eval_bw_res(self) -> List[float]:
        return self.eval_bw_res

    def get_orders(self) -> List[int]:
        return getattr(self, "current_orders", [])

    def get_inventory(self) -> List[int]:
        return self.inventory

    def get_demand_history(self) -> List[List[int]]:
        return self.demand_history

    def get_fulfilled_history(self) -> List[List[int]]:
        return self.fulfilled_history

    def get_inventory_history(self) -> List[List[int]]:
        return self.inventory_history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _action_map(self, actions: List[int]) -> List[int]:
        mapped_actions = [min(max(int(a), 0), self.action_dim - 1) for a in actions]
        self.current_orders = mapped_actions
        return mapped_actions

    def _get_reset_obs(self) -> List[np.ndarray]:
        obs_list: List[np.ndarray] = []
        for i in range(self.level_num):
            inv = self.inventory[i]
            back = self.backlog[i]
            down_info = self.init_outstanding
            pipeline = self.pipeline_orders[i]
            arr = np.array([inv, back, down_info] + pipeline, dtype=float)
            if self.normalize:
                arr = arr / (self.action_dim - 1)
            obs_list.append(arr.reshape(self.obs_dim))
        return obs_list

    def _get_step_obs(self, actions: List[int]) -> List[np.ndarray]:
        obs_list: List[np.ndarray] = []
        # Downstream agent (0) observes customer demand directly
        downstream_dem = self.demand_list[self.step_num - 1]
        inv = self.inventory[0]
        back = self.backlog[0]
        pipe = self.pipeline_orders[0]
        arr = np.array([inv, back, downstream_dem] + pipe, dtype=float)
        if self.normalize:
            arr = arr / (self.action_dim - 1)
        obs_list.append(arr.reshape(self.obs_dim))
        # Upstream agents observe previous agent's order instead of demand
        for i in range(1, self.level_num):
            inv_i = self.inventory[i]
            back_i = self.backlog[i]
            down_action = actions[i - 1]
            pipe_i = self.pipeline_orders[i]
            arr = np.array([inv_i, back_i, down_action] + pipe_i, dtype=float)
            if self.normalize:
                arr = arr / (self.action_dim - 1)
            obs_list.append(arr.reshape(self.obs_dim))
        return obs_list

    def _get_processed_rewards(self, rewards: List[float]) -> List[List[float]]:
        if self.train:
            mean_r = float(np.mean(rewards))
            processed = [[self.alpha * r + (1.0 - self.alpha) * mean_r] for r in rewards]
            return processed
        else:
            return [[r] for r in rewards]

    def _state_update(self, actions: List[int]) -> List[float]:
        self.action_history = [h + [a] for h, a in zip(self.action_history, actions)]
        
        downstream_demands: List[int] = [self.demand_list[self.step_num]] + actions[:-1]
        
        for i, d in enumerate(downstream_demands):
            self.demand_history[i].append(d)
        
        effective_demand = [d + self.backlog[i] for i, d in enumerate(downstream_demands)]
        
        # Shipping loss disabled
        random_shipping_loss = False
        lost_rate = [1.0 for _ in range(self.level_num)]
        if random_shipping_loss:
            lost_rate = [1.0 - random.random() * 0.1 for _ in range(self.level_num)]
        
        self.step_num += 1
        rewards: List[float] = []
        
        for i in range(self.level_num):
            received = int(self.pipeline_orders[i][0] * lost_rate[i])
            available = self.inventory[i] + received
            
            fulfilled = min(effective_demand[i], available)
            self.fulfilled_history[i].append(fulfilled)
            
            unmet = effective_demand[i] - available
            
            if unmet > 0:
                self.backlog[i] = unmet
                self.inventory[i] = 0
            else:
                self.backlog[i] = 0
                self.inventory[i] = -unmet
            self.backlog_history[i].append(self.backlog[i])
            
            self.inventory_history[i].append(self.inventory[i])
            
            new_order = actions[i]
            self.pipeline_orders[i].append(new_order)
            self.pipeline_orders[i].pop(0)
            
            order_cost_var = 0.0
            if self.price_discount and actions[i] > 0:
                bucket_idx = min(actions[i] // 5, len(self.discount_schedule) - 1)
                order_cost_var = self.discount_schedule[bucket_idx] * actions[i]
            
            order_cost_fix = self.fixed_cost if actions[i] > 0 else 0.0
            
            cost = (
                self.inventory[i] * self.holding_cost[i]
                + self.backlog[i] * self.backlog_cost[i]
                + order_cost_var
                + order_cost_fix
            )
            rewards.append(-cost)
        
        if not self.train:
            if self.step_num == self.episode_len:
                for k in range(self.level_num):
                    hist = self.action_history[k]
                    if not hist or np.mean(hist) < 1e-6:
                        self.record_act_sta[k].append(0.0)
                    else:
                        cv = float(np.std(hist) / np.mean(hist))
                        self.record_act_sta[k].append(cv)
                if self.eval_index == 0:
                    self.eval_bw_res = [float(np.mean(sta)) for sta in self.record_act_sta]
                    self.record_act_sta = [[] for _ in range(self.level_num)]
        return rewards

    # --- FIX: ADDED CLOSE METHOD ---
    def close(self):
        """Clean up resources (required by VecEnv)."""
        pass
    # -------------------------------