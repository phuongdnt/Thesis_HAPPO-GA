"""
happo_agent.py
===============
FIXED VERSION: Added Learning Rate Scheduler (StepLR) to fix non-convergence.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR 

from agents.policy_networks import MLPActor
from agents.centralized_critic import CentralizedCritic
from agents.replay_buffer import ReplayBuffer


class HAPPOAgent:
    """Multi–agent PPO with heterogeneous policies and a central critic."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: Optional[str] = None,
    ) -> None:
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create actor networks for each agent
        self.actors: List[MLPActor] = []
        self.actor_optimisers: List[Adam] = []
        for _ in range(num_agents):
            actor = MLPActor(obs_dim, action_dim, hidden_dim).to(self.device)
            self.actors.append(actor)
            self.actor_optimisers.append(Adam(actor.parameters(), lr=actor_lr))
            
        # Centralised critic
        state_dim = num_agents * obs_dim
        self.critic = CentralizedCritic(state_dim, critic_hidden_dim)
        self.critic_net = self.critic.net.to(self.device)
        self.critic_optimizer = Adam(self.critic_net.parameters(), lr=critic_lr)
        
        # --- FIX: Initialize LR Schedulers ---
        # Reduce LR by 10% every 1000 update steps for better convergence
        self.actor_schedulers = [StepLR(opt, step_size=1000, gamma=0.9) for opt in self.actor_optimisers]
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=1000, gamma=0.9)
        # -------------------------------------

        self.buffer = ReplayBuffer()

    def select_actions(self, obs_list: List[np.ndarray]) -> Tuple[List[int], List[torch.Tensor]]:
        actions: List[int] = []
        log_probs: List[torch.Tensor] = []
        for i, obs in enumerate(obs_list):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            act, log_prob = self.actors[i].get_action(obs_tensor)
            actions.append(act)
            log_probs.append(log_prob)
        return actions, log_probs

    def store_transition(
        self,
        obs: List[np.ndarray],
        actions: List[int],
        log_probs: List[torch.Tensor],
        rewards: List[float],
        next_obs: List[np.ndarray],
        done: bool,
    ) -> None:
        log_probs_detached = [float(lp.detach().cpu().item()) for lp in log_probs]
        transition = {
            "obs": obs,
            "actions": actions,
            "log_probs": log_probs_detached,
            "rewards": rewards,
            "next_obs": next_obs,
            "done": done,
        }
        self.buffer.add(transition)

    def update(self, batch_size: Optional[int] = None) -> None:
        if len(self.buffer) == 0:
            return
        data = self.buffer.as_dict()
        T = len(self.buffer)
        
        obs_tensor = torch.tensor(data["obs"], dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(data["actions"], dtype=torch.int64, device=self.device)
        old_log_probs_tensor = torch.tensor(data["log_probs"], dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(data["rewards"], dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.tensor(data["next_obs"], dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(data["done"], dtype=torch.float32, device=self.device)
        if dones_tensor.dim() > 1:
            dones_tensor = dones_tensor.max(dim=1).values
            
        states = obs_tensor.view(T, -1)
        next_states = next_obs_tensor.view(T, -1)
        
        with torch.no_grad():
            values = self.critic_net(states).squeeze(-1)
            next_values = self.critic_net(next_states).squeeze(-1)
            
        global_rewards = rewards_tensor.sum(dim=1)
        advantages = torch.zeros(T, device=self.device)
        returns = torch.zeros(T, device=self.device)
        gae = 0.0
        
        for t in reversed(range(T)):
            mask = 1.0 - dones_tensor[t]
            delta = global_rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update Critic
        self.critic_optimizer.zero_grad()
        value_preds = self.critic_net(states).squeeze(-1)
        critic_loss = torch.mean((returns - value_preds) ** 2)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        # Update Actors
        for i in range(self.num_agents):
            actor = self.actors[i]
            actor_opt = self.actor_optimisers[i]
            
            obs_i = obs_tensor[:, i, :]
            actions_i = actions_tensor[:, i]
            old_log_probs_i = old_log_probs_tensor[:, i]
            
            new_log_probs, entropy = actor.evaluate_actions(obs_i, actions_i)
            ratio = torch.exp(new_log_probs - old_log_probs_i.to(self.device))
            
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            actor_loss = -torch.mean(torch.min(surrogate1, surrogate2)) - self.entropy_coef * torch.mean(entropy)
            
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            actor_opt.step()
            
        # --- FIX: Step Schedulers ---
        for sch in self.actor_schedulers:
            sch.step()
        self.critic_scheduler.step()
        # ----------------------------
        
        self.buffer.clear()
