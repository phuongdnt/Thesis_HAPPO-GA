"""
reward_utils.py - Multi-Objective Reward for HAPPO + GA
========================================================
Simple reward function for Pareto multi-objective optimization.
No inventory bonus, no hierarchy penalty, no hacking.
"""

from typing import List
import numpy as np


def compute_mo_reward(
    raw_rewards: List[float],
    fill_rates: List[float],
    cost_weight: float,
    service_weight: float,
    service_bonus_scale: float = 10.0,
) -> List[float]:
    """
    Multi-objective reward function.
    
    Reward_i = cost_weight × raw_reward_i + service_weight × fill_rate_i × scale
    
    Args:
        raw_rewards: Per-agent rewards (negative costs)
        fill_rates: Per-agent fill rates [0-1]
        cost_weight: Weight for cost objective
        service_weight: Weight for service objective
        service_bonus_scale: Scaling factor for service component
    
    Returns:
        List of rewards per agent
    """
    mo_rewards = []
    
    for raw_r, fr in zip(raw_rewards, fill_rates):
        cost_component = cost_weight * raw_r
        service_component = service_weight * fr * service_bonus_scale
        mo_rewards.append(cost_component + service_component)
    
    return mo_rewards


def compute_cycle_service_level(backlog_history: List[List[float]]) -> List[float]:
    """
    Compute cycle service level per agent.
    CSL = proportion of periods with zero backlog.
    """
    if not backlog_history or not backlog_history[0]:
        return [1.0]
    
    num_agents = len(backlog_history)
    csl_per_agent = []
    
    for agent_backlogs in backlog_history:
        if len(agent_backlogs) == 0:
            csl_per_agent.append(1.0)
        else:
            zero_backlog_periods = sum(1 for b in agent_backlogs if b == 0)
            csl_per_agent.append(zero_backlog_periods / len(agent_backlogs))
    
    return csl_per_agent
