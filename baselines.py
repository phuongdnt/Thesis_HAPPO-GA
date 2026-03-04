"""
baselines.py - Classical Inventory Baselines
=============================================
Baselines để so sánh với HAPPO + GA Hybrid:
1. Base-stock policy (order-up-to level)
2. (s,S) policy (reorder point, order-up-to)

Cả 2 đều là policies kinh điển trong inventory management.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class BaseStockPolicy:
    """
    Base-stock (Order-up-to) Policy.
    
    Mỗi period, order để đưa inventory position lên target level S.
    Order quantity = S - (inventory + outstanding orders)
    
    Optimal S = mean_demand × (lead_time + 1) + safety_stock
    """
    target_levels: List[float]  # S for each agent
    
    def get_action(self, inventory: float, outstanding: float, agent_idx: int) -> int:
        """Compute order quantity."""
        S = self.target_levels[agent_idx]
        inventory_position = inventory + outstanding
        order = max(0, S - inventory_position)
        return int(round(order))
    
    def get_actions(self, inventories: List[float], outstandings: List[float]) -> List[int]:
        """Get actions for all agents."""
        return [self.get_action(inv, out, i) 
                for i, (inv, out) in enumerate(zip(inventories, outstandings))]


@dataclass  
class sSPolicy:
    """
    (s, S) Policy - Reorder Point Policy.
    
    If inventory position <= s: order up to S
    Else: don't order
    
    s = reorder point (trigger)
    S = order-up-to level
    """
    reorder_points: List[float]  # s for each agent
    target_levels: List[float]   # S for each agent
    
    def get_action(self, inventory: float, outstanding: float, agent_idx: int) -> int:
        """Compute order quantity."""
        s = self.reorder_points[agent_idx]
        S = self.target_levels[agent_idx]
        inventory_position = inventory + outstanding
        
        if inventory_position <= s:
            order = S - inventory_position
        else:
            order = 0
        
        return int(max(0, round(order)))
    
    def get_actions(self, inventories: List[float], outstandings: List[float]) -> List[int]:
        """Get actions for all agents."""
        return [self.get_action(inv, out, i) 
                for i, (inv, out) in enumerate(zip(inventories, outstandings))]


def compute_optimal_base_stock_levels(
    mean_demand: float,
    std_demand: float,
    lead_time: int,
    holding_cost: List[float],
    backlog_cost: List[float],
    service_level_target: float = 0.95,
) -> List[float]:
    """
    Compute optimal base-stock levels using newsvendor logic.
    
    S* = μ(L+1) + z × σ × √(L+1)
    
    where z = Φ^(-1)(b / (b + h)) for cost-optimal
    or z = Φ^(-1)(service_level) for service-level target
    """
    from scipy.stats import norm
    
    n_agents = len(holding_cost)
    levels = []
    
    for i in range(n_agents):
        h = holding_cost[i]
        b = backlog_cost[i]
        
        # Critical ratio
        critical_ratio = b / (b + h)
        z = norm.ppf(min(critical_ratio, service_level_target))
        
        # Base-stock level
        review_period = lead_time + 1
        S = mean_demand * review_period + z * std_demand * np.sqrt(review_period)
        levels.append(max(0, S))
    
    return levels


def compute_optimal_sS_params(
    mean_demand: float,
    std_demand: float,
    lead_time: int,
    holding_cost: List[float],
    backlog_cost: List[float],
    fixed_cost: float = 1.0,
) -> Tuple[List[float], List[float]]:
    """
    Compute (s, S) policy parameters.
    
    Simplified approach:
    - s = reorder point = mean_demand × lead_time + safety_stock
    - S = s + EOQ
    
    EOQ = sqrt(2 × K × D / h) where K = fixed cost, D = demand, h = holding
    """
    from scipy.stats import norm
    
    n_agents = len(holding_cost)
    reorder_points = []
    target_levels = []
    
    for i in range(n_agents):
        h = holding_cost[i]
        b = backlog_cost[i]
        
        # Safety factor
        service_level = b / (b + h)
        z = norm.ppf(min(0.99, service_level))
        
        # Reorder point
        s = mean_demand * lead_time + z * std_demand * np.sqrt(lead_time)
        
        # EOQ-based order quantity
        annual_demand = mean_demand * 100  # Assume 100 periods
        Q = np.sqrt(2 * fixed_cost * annual_demand / h)
        
        # Order-up-to level
        S = s + Q
        
        reorder_points.append(max(0, s))
        target_levels.append(max(s + 1, S))
    
    return reorder_points, target_levels


def evaluate_baseline_policy(
    env,
    policy,
    n_episodes: int = 30,
    policy_name: str = "baseline"
) -> Dict:
    """
    Evaluate a baseline policy on the environment.
    
    Returns metrics dict compatible with HAPPO evaluation format.
    """
    from utils.metrics import compute_bullwhip, compute_service_levels
    
    all_costs = []
    all_fill_rates = []
    all_bullwhips = []
    all_csl = []
    
    num_agents = env.agent_num
    
    for ep in range(n_episodes):
        obs = env.reset(train=False)
        orders = [[] for _ in range(num_agents)]
        ep_costs = [0.0] * num_agents
        
        while True:
            # Get current state
            inventories = [env.inventory[i] for i in range(num_agents)]
            outstandings = [sum(env.pipeline_orders[i]) for i in range(num_agents)]
            
            # Get actions from policy
            actions = policy.get_actions(inventories, outstandings)
            
            for i, a in enumerate(actions):
                orders[i].append(a)
            
            # Step environment
            obs, rewards, done, infos = env.step(actions, one_hot=False)
            
            # Accumulate costs
            for i, r in enumerate(rewards):
                cost = -r[0] if isinstance(r, list) else -r
                ep_costs[i] += cost
            
            if all(done):
                break
        
        all_costs.append(ep_costs)
        all_bullwhips.append(compute_bullwhip(orders))
        all_fill_rates.append(compute_service_levels(
            env.get_demand_history(), 
            env.get_fulfilled_history()
        ))
        
        # Cycle service level
        csl = []
        for agent_backlogs in env.backlog_history:
            if agent_backlogs:
                zero_periods = sum(1 for b in agent_backlogs if b == 0)
                csl.append(zero_periods / len(agent_backlogs))
            else:
                csl.append(1.0)
        all_csl.append(csl)
    
    # Aggregate
    per_agent_cost = [np.mean([c[i] for c in all_costs]) for i in range(num_agents)]
    per_agent_fr = [np.mean([f[i] for f in all_fill_rates]) for i in range(num_agents)]
    per_agent_bw = [np.mean([b[i] for b in all_bullwhips]) for i in range(num_agents)]
    per_agent_csl = [np.mean([s[i] for s in all_csl]) for i in range(num_agents)]
    
    return {
        "policy_name": policy_name,
        "total_cost": float(sum(per_agent_cost)),
        "cost_std": float(np.std([sum(c) for c in all_costs])),
        "fill_rate_mean": float(np.mean(per_agent_fr)),
        "service_level_mean": float(np.mean(per_agent_csl)),
        "bullwhip_mean": float(np.mean(per_agent_bw)),
        "cost_per_agent": [float(c) for c in per_agent_cost],
        "fill_rate_per_agent": [float(f) for f in per_agent_fr],
        "service_level_per_agent": [float(s) for s in per_agent_csl],
        "bullwhip_per_agent": [float(b) for b in per_agent_bw],
    }


def run_baseline_comparison(env, mean_demand: float = 10.0, std_demand: float = 3.0):
    """
    Run comparison of baseline policies.
    
    Usage:
        from envs.serial_env import SerialInventoryEnv
        env = SerialInventoryEnv(...)
        results = run_baseline_comparison(env)
    """
    lead_time = env.lead_time if hasattr(env, 'lead_time') else 2
    holding_cost = env.holding_cost if hasattr(env, 'holding_cost') else [1.0] * env.agent_num
    backlog_cost = env.backlog_cost if hasattr(env, 'backlog_cost') else [5.0, 3.0, 2.0]
    
    results = {}
    
    # 1. Base-stock policy
    print("Evaluating Base-stock policy...")
    base_stock_levels = compute_optimal_base_stock_levels(
        mean_demand, std_demand, lead_time,
        holding_cost, backlog_cost
    )
    base_stock_policy = BaseStockPolicy(target_levels=base_stock_levels)
    results["base_stock"] = evaluate_baseline_policy(
        env, base_stock_policy, n_episodes=30, policy_name="Base-stock"
    )
    print(f"  Base-stock levels: {[f'{s:.1f}' for s in base_stock_levels]}")
    
    # 2. (s,S) policy
    print("Evaluating (s,S) policy...")
    s_points, S_levels = compute_optimal_sS_params(
        mean_demand, std_demand, lead_time,
        holding_cost, backlog_cost
    )
    sS_policy = sSPolicy(reorder_points=s_points, target_levels=S_levels)
    results["sS"] = evaluate_baseline_policy(
        env, sS_policy, n_episodes=30, policy_name="(s,S)"
    )
    print(f"  (s,S) params: s={[f'{s:.1f}' for s in s_points]}, S={[f'{S:.1f}' for S in S_levels]}")
    
    return results


if __name__ == "__main__":
    # Quick test
    print("Testing baseline policies...")
    
    # Test BaseStockPolicy
    policy = BaseStockPolicy(target_levels=[30.0, 25.0, 20.0])
    actions = policy.get_actions([10, 15, 20], [5, 5, 0])
    print(f"Base-stock actions: {actions}")
    
    # Test sSPolicy
    policy = sSPolicy(reorder_points=[15, 12, 10], target_levels=[35, 30, 25])
    actions = policy.get_actions([10, 15, 20], [5, 5, 0])
    print(f"(s,S) actions: {actions}")
    
    print("✅ Baseline policies test passed")
