"""
train_pareto_network.py - Network Topology Pareto Training
===========================================================
HAPPO + GA Hybrid with Multi-Objective Pareto Analysis

Usage: python train_pareto_network.py --config configs/pareto_network.yaml
"""

from __future__ import annotations
import argparse, os, json, yaml, time
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import numpy as np
import torch

from utils.logger import setup_logger
from agents.happo_agent import HAPPOAgent
from envs.network_env import NetworkInventoryEnv
from envs.vec_env import SubprocVecEnv
from lot_sizing.hybrid_planner import HybridPlanner
from utils.metrics import compute_bullwhip, compute_service_levels
from training_utils import (
    EarlyStoppingConfig, EarlyStoppingTracker, RewardSmoother, apply_lr_factor
)
from reward_utils import compute_mo_reward, compute_cycle_service_level


# Whitelist of valid NetworkInventoryEnv parameters
NETWORK_ENV_PARAMS = {
    'lead_time', 'episode_len', 'action_dim', 'init_inventory', 'init_outstanding',
    'holding_cost', 'backlog_cost', 'fixed_cost', 'external_demand_dist',
    'external_max_demand', 'eval_data_dirs', 'rng_seed'
}


class ParetoLogger:
    """Logger for Pareto experiment tracking."""
    
    def __init__(self, output_dir: str, name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / f"{name}_log.json"
        self.data = {
            "name": name, 
            "start": datetime.now().isoformat(), 
            "history": [], 
            "evals": [], 
            "early_stopping": {}
        }
    
    def log_config(self, cfg): 
        self.data["config"] = cfg
        self._save()
    
    def log_episode(self, ep, metrics): 
        self.data["history"].append({"ep": ep, **metrics})
        if ep % 10 == 0: 
            self._save()
    
    def log_eval(self, ep, metrics): 
        self.data["evals"].append({"ep": ep, **metrics})
        self._save()
    
    def log_early_stopping(self, summary): 
        self.data["early_stopping"] = summary
        self._save()
    
    def finalize(self): 
        self.data["end"] = datetime.now().isoformat()
        self._save()
    
    def _save(self):
        with open(self.log_file, 'w') as f: 
            json.dump(self.data, f, indent=2, default=str)


def parse_config(path: str) -> Dict:
    """Parse YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Resolve eval_data_dirs paths
    env_cfg = cfg.get("env", {})
    if "eval_data_dirs" in env_cfg:
        base = os.path.dirname(os.path.abspath(path))
        resolved = []
        for d in env_cfg["eval_data_dirs"]:
            if not os.path.isabs(d):
                for b in [os.getcwd(), base]:
                    if os.path.exists(os.path.join(b, d)):
                        resolved.append(os.path.abspath(os.path.join(b, d)))
                        break
                else:
                    resolved.append(d)
            else:
                resolved.append(d)
        env_cfg["eval_data_dirs"] = resolved
    
    return cfg


def build_network_env(cfg: Dict) -> NetworkInventoryEnv:
    """Build network inventory environment."""
    env_cfg = cfg.get("env", {})
    children = {int(k): [int(x) for x in v] for k, v in env_cfg.get("children", {}).items()}
    parents = {int(k): (int(v) if v else None) for k, v in env_cfg.get("parents", {}).items()}
    env_params = {k: v for k, v in env_cfg.items() if k in NETWORK_ENV_PARAMS}
    return NetworkInventoryEnv(children=children, parents=parents, **env_params)


def run_evaluation(env, agent, planner, n_episodes: int = 30) -> Dict:
    """Run evaluation episodes and compute metrics."""
    all_costs, all_bw, all_fr, all_csl = [], [], [], []
    num_agents = env.agent_num
    
    for _ in range(n_episodes):
        obs = env.reset(train=False)
        orders = [[] for _ in range(num_agents)]
        ep_rewards = []
        
        while True:
            actions, _ = agent.select_actions(obs)
            if planner:
                actions = planner.refine_actions(actions)
            
            for i, a in enumerate(actions):
                orders[i].append(a)
            
            obs, rewards, done, _ = env.step(actions, one_hot=False)
            ep_rewards.append([r[0] if isinstance(r, list) else r for r in rewards])
            
            if all(done):
                break
        
        # Compute metrics
        costs = [-sum(ep_rewards[t][i] for t in range(len(ep_rewards))) for i in range(num_agents)]
        all_costs.append(costs)
        all_bw.append(compute_bullwhip(orders))
        all_fr.append(compute_service_levels(env.get_demand_history(), env.get_fulfilled_history()))
        
        # Cycle service level
        csl = compute_cycle_service_level(env.backlog_history)
        all_csl.append(csl)
    
    # Aggregate results
    per_agent_cost = [np.mean([c[i] for c in all_costs]) for i in range(num_agents)]
    per_agent_fr = [np.mean([f[i] for f in all_fr]) for i in range(num_agents)]
    per_agent_bw = [np.mean([b[i] for b in all_bw]) for i in range(num_agents)]
    per_agent_csl = [np.mean([s[i] for s in all_csl]) for i in range(num_agents)]
    
    # Echelon averages for network
    retailer_avg_fr = (per_agent_fr[0] + per_agent_fr[1]) / 2
    distributor_avg_fr = (per_agent_fr[2] + per_agent_fr[3]) / 2
    factory_avg_fr = (per_agent_fr[4] + per_agent_fr[5]) / 2
    
    return {
        "total_cost": float(sum(per_agent_cost)),
        "cost_std": float(np.std([sum(c) for c in all_costs])),
        "fill_rate_mean": float(np.mean(per_agent_fr)),
        "service_level_mean": float(np.mean(per_agent_csl)),
        "bullwhip_mean": float(np.mean(per_agent_bw)),
        "cost_per_agent": [float(c) for c in per_agent_cost],
        "fill_rate_per_agent": [float(f) for f in per_agent_fr],
        "service_level_per_agent": [float(s) for s in per_agent_csl],
        "bullwhip_per_agent": [float(b) for b in per_agent_bw],
        "retailer_avg_fr": retailer_avg_fr,
        "distributor_avg_fr": distributor_avg_fr,
        "factory_avg_fr": factory_avg_fr,
    }


def print_eval_metrics(metrics: Dict, num_agents: int, agent_names: List[str], logger):
    """Print evaluation metrics in table format."""
    logger.info(f"  {'Agent':<14} {'Cost':<10} {'FillRate':<10} {'ServiceLv':<10} {'Bullwhip':<10}")
    logger.info(f"  {'-'*54}")
    
    for i in range(num_agents):
        name = agent_names[i] if i < len(agent_names) else f"Agent_{i}"
        logger.info(f"  {name:<14} {metrics['cost_per_agent'][i]:<10.1f} "
                   f"{metrics['fill_rate_per_agent'][i]*100:<10.2f}% "
                   f"{metrics['service_level_per_agent'][i]*100:<10.2f}% "
                   f"{metrics['bullwhip_per_agent'][i]:<10.3f}")
    
    logger.info(f"  {'-'*54}")
    logger.info(f"  {'TOTAL':<14} {metrics['total_cost']:<10.1f} "
               f"{metrics['fill_rate_mean']*100:<10.2f}% "
               f"{metrics['service_level_mean']*100:<10.2f}% "
               f"{metrics['bullwhip_mean']:<10.3f}")
    
    # Echelon summary
    logger.info(f"  Echelon FR: R={metrics['retailer_avg_fr']*100:.1f}% | "
               f"D={metrics['distributor_avg_fr']*100:.1f}% | "
               f"F={metrics['factory_avg_fr']*100:.1f}%")


def train_single_config(cfg: Dict, cw: float, sw: float, seed: int, 
                        output_dir: str, logger) -> Dict:
    """Train single configuration."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    name = f"net_cw{cw:.1f}_sw{sw:.1f}_s{seed}"
    exp_log = ParetoLogger(output_dir, name)
    exp_log.log_config({"cost_weight": cw, "service_weight": sw, "seed": seed})
    
    logger.info(f"\n{'='*60}")
    logger.info(f"[NETWORK] cost_weight={cw}, service_weight={sw}, seed={seed}")
    logger.info(f"{'='*60}")
    
    # Build environment
    base_env = build_network_env(cfg)
    num_agents = base_env.agent_num
    agent_names = ["Retailer_1", "Retailer_2", "Distributor_1", "Distributor_2", "Factory_1", "Factory_2"]
    
    env_cfg = cfg.get("env", {})
    service_bonus_scale = env_cfg.get("service_bonus_scale", 10.0)
    
    logger.info(f"[CONFIG] Backlog costs: {env_cfg.get('backlog_cost', [5,5,3,3,2,2])}")
    
    # Build agent
    agent_cfg = cfg.get("agent", {})
    agent = HAPPOAgent(
        obs_dim=base_env.obs_dim,
        action_dim=env_cfg.get("action_dim", 41),
        num_agents=num_agents,
        hidden_dim=agent_cfg.get("hidden_dim", 128),
        critic_hidden_dim=agent_cfg.get("critic_hidden_dim", 256),
        actor_lr=agent_cfg.get("actor_lr", 1e-4),
        critic_lr=agent_cfg.get("critic_lr", 1e-4),
        gamma=agent_cfg.get("gamma", 0.99),
        gae_lambda=agent_cfg.get("gae_lambda", 0.95),
        eps_clip=agent_cfg.get("eps_clip", 0.2),
        value_coef=agent_cfg.get("value_coef", 0.5),
        entropy_coef=agent_cfg.get("entropy_coef", 0.05),
    )
    
    # Training config
    train_cfg = cfg.get("training", {})
    episodes = train_cfg.get("episodes", 8000)
    eval_every = train_cfg.get("evaluate_every", 100)
    n_threads = train_cfg.get("n_rollout_threads", 8)
    
    # Early stopping
    es_config = EarlyStoppingConfig(
        enabled=train_cfg.get("early_stop", True),
        patience=train_cfg.get("n_no_improvement_thres", 20),
        min_delta=train_cfg.get("min_improvement_delta", 0.01),
        warmup_evaluations=train_cfg.get("n_warmup_evaluations", 15),
        reduce_lr_on_plateau=train_cfg.get("reduce_lr_on_plateau", True),
        lr_reduction_factor=train_cfg.get("lr_reduction_factor", 0.5),
        lr_reduction_patience=train_cfg.get("lr_reduction_patience", 5),
    )
    early_stopper = EarlyStoppingTracker(es_config, logger)
    reward_smoother = RewardSmoother(alpha=0.1, window=50)
    
    # Vectorized environment
    if n_threads > 1:
        env = SubprocVecEnv([lambda: build_network_env(cfg) for _ in range(n_threads)])
        is_vec = True
    else:
        env = base_env
        is_vec = False
    
    # GA planner for evaluation
    eval_planner = None
    if train_cfg.get("use_ga", True):
        ga_cfg = cfg.get("heuristic", {}).get("ga", {})
        eval_planner = HybridPlanner(
            env=base_env, 
            horizon=train_cfg.get("ga_horizon", 5),
            use_ga=True, 
            ga_params=ga_cfg
        )
    
    start_time = time.time()
    stop_reason = "max_episodes"
    checkpoint_every = train_cfg.get("checkpoint_every", 500)
    
    # Training loop
    for ep in range(1, episodes + 1):
        ep_reward = 0.0
        
        if is_vec:
            obs_batch = env.reset(train=True)
            done_batch = np.zeros((n_threads, num_agents), dtype=bool)
            
            while not np.all(done_batch):
                acts_batch, lps_batch = [], []
                for i in range(n_threads):
                    if not np.all(done_batch[i]):
                        obs_i = obs_batch[i].tolist() if isinstance(obs_batch[i], np.ndarray) else obs_batch[i]
                        a, lp = agent.select_actions(obs_i)
                        acts_batch.append(a)
                        lps_batch.append(lp)
                    else:
                        acts_batch.append([0] * num_agents)
                        lps_batch.append([0.0] * num_agents)
                
                next_obs, rews, done2, infos = env.step(acts_batch, one_hot=False)
                
                for i in range(n_threads):
                    if not np.all(done_batch[i]):
                        obs_i = obs_batch[i].tolist() if isinstance(obs_batch[i], np.ndarray) else obs_batch[i]
                        next_i = next_obs[i].tolist() if isinstance(next_obs[i], np.ndarray) else next_obs[i]
                        rews_i = rews[i].tolist() if isinstance(rews[i], np.ndarray) else rews[i]
                        done_i = done2[i].tolist() if isinstance(done2[i], np.ndarray) else done2[i]
                        flat_r = [r[0] if isinstance(r, list) else r for r in rews_i]
                        
                        # Get fill rates from info
                        info_i = infos[i] if isinstance(infos[i], list) else [{} for _ in range(num_agents)]
                        frs = [inf.get('sales', 0) / inf.get('demand', 1) 
                               if inf.get('demand', 0) > 1e-6 else 1.0 for inf in info_i]
                        
                        # Compute MO reward
                        mo_r = compute_mo_reward(flat_r, frs, cw, sw, service_bonus_scale)
                        
                        agent.store_transition(obs_i, acts_batch[i], lps_batch[i], mo_r, next_i, done_i)
                        ep_reward += sum(mo_r)
                
                obs_batch = next_obs
                done_batch = done2
        else:
            obs = env.reset(train=True)
            while True:
                acts, lps = agent.select_actions(obs)
                next_obs, rews, done, infos = env.step(acts, one_hot=False)
                flat_r = [r[0] if isinstance(r, list) else r for r in rews]
                
                info_list = infos if isinstance(infos, list) else [{} for _ in range(num_agents)]
                frs = [inf.get('sales', 0) / inf.get('demand', 1) 
                       if inf.get('demand', 0) > 1e-6 else 1.0 for inf in info_list]
                
                mo_r = compute_mo_reward(flat_r, frs, cw, sw, service_bonus_scale)
                
                agent.store_transition(obs, acts, lps, mo_r, next_obs, done[0])
                ep_reward += sum(mo_r)
                obs = next_obs
                if all(done):
                    break
        
        # Update agent
        agent.update()
        smoothed = reward_smoother.update(ep_reward)
        
        # Logging
        if ep % 50 == 0:
            logger.info(f"[NET] Ep {ep}/{episodes}: reward={ep_reward:.1f}, EMA={smoothed:.1f}")
            exp_log.log_episode(ep, {"reward": ep_reward, "ema": smoothed})
        
        # Checkpoint saving
        if checkpoint_every and ep % checkpoint_every == 0:
            checkpoint_path = os.path.join(output_dir, f"{name}_checkpoint_ep{ep}.pth")
            torch.save({
                "actors": [a.state_dict() for a in agent.actors],
                "critic": agent.critic_net.state_dict(),
                "episode": ep,
                "config": {"cost_weight": cw, "service_weight": sw, "seed": seed},
            }, checkpoint_path)
            logger.info(f"  [CHECKPOINT] Saved to {checkpoint_path}")
        
        # Evaluation
        if eval_every and ep % eval_every == 0:
            metrics = run_evaluation(base_env, agent, eval_planner, 10)
            
            # Score for early stopping
            score = -cw * metrics["total_cost"] + sw * metrics["fill_rate_mean"] * 1000
            metrics["score"] = score
            
            model_state = {
                "actors": [a.state_dict() for a in agent.actors],
                "critic": agent.critic_net.state_dict()
            }
            
            should_stop, reason, lr_factor = early_stopper.update(ep, metrics, model_state)
            if lr_factor:
                apply_lr_factor(agent, lr_factor)
            
            improve_marker = "★ BEST" if reason == "improved" else ""
            logger.info(f"\n  {'─'*58}")
            logger.info(f"  [EVAL] Episode {ep} {improve_marker}")
            logger.info(f"  {'─'*58}")
            print_eval_metrics(metrics, num_agents, agent_names, logger)
            
            if es_config.enabled:
                es = early_stopper.get_summary()
                logger.info(f"  [ES] Best@Ep{es['best_episode']}: {es['best_score']:.1f} | "
                           f"NoImprove: {es['no_improve_streak']}/{es_config.patience}")
            logger.info(f"  {'─'*58}")
            
            exp_log.log_eval(ep, metrics)
            
            if should_stop:
                stop_reason = reason
                logger.info(f"\n[EARLY STOP] {reason}")
                break
    
    train_time = time.time() - start_time
    
    # Load best model
    if early_stopper.best_model_state:
        for a, sd in zip(agent.actors, early_stopper.best_model_state["actors"]):
            a.load_state_dict(sd)
        agent.critic_net.load_state_dict(early_stopper.best_model_state["critic"])
    
    # Final evaluation
    logger.info(f"\n{'='*60}\nFINAL EVALUATION (30 episodes)\n{'='*60}")
    final = run_evaluation(base_env, agent, eval_planner, 30)
    final["train_time"] = train_time
    final["episodes_trained"] = ep
    final["stop_reason"] = stop_reason
    
    print_eval_metrics(final, num_agents, agent_names, logger)
    logger.info(f"\n[DONE] Time: {train_time/60:.1f}min, Stop: {stop_reason}")
    
    # Save model
    torch.save({
        "actors": [a.state_dict() for a in agent.actors],
        "critic": agent.critic_net.state_dict(),
        "config": {"cost_weight": cw, "service_weight": sw, "seed": seed},
        "metrics": final
    }, os.path.join(output_dir, f"{name}.pth"))
    
    exp_log.log_early_stopping(early_stopper.get_summary())
    exp_log.finalize()
    
    if is_vec:
        env.close()
    
    return final


def main(config_path: str):
    """Main training loop."""
    logger = setup_logger()
    cfg = parse_config(config_path)
    
    pareto = cfg.get("pareto", {})
    weights = pareto.get("weight_configs", [
        {"name": "balanced", "cost_weight": 0.5, "service_weight": 0.5}
    ])
    seeds = pareto.get("seeds", [1, 2, 3])
    output = pareto.get("output_dir", "results/pareto_network")
    os.makedirs(output, exist_ok=True)
    
    results = []
    
    logger.info("=" * 60)
    logger.info("PARETO FRONTIER - NETWORK TOPOLOGY")
    logger.info(f"Configs: {len(weights)} | Seeds: {len(seeds)} | Total: {len(weights)*len(seeds)}")
    logger.info("=" * 60)
    
    for wc in weights:
        cw, sw = wc["cost_weight"], wc["service_weight"]
        name = wc.get("name", f"cw{cw}_sw{sw}")
        cfg_results = []
        
        for seed in seeds:
            try:
                m = train_single_config(cfg, cw, sw, seed, output, logger)
                m.update({
                    "config_name": name, 
                    "cost_weight": cw, 
                    "service_weight": sw, 
                    "seed": seed
                })
                cfg_results.append(m)
                results.append(m)
            except Exception as e:
                logger.error(f"Failed {name} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
        
        if cfg_results:
            logger.info(f"\n[{name}] Avg Cost={np.mean([r['total_cost'] for r in cfg_results]):.1f}, "
                       f"FR={np.mean([r['fill_rate_mean'] for r in cfg_results])*100:.2f}%, "
                       f"SL={np.mean([r['service_level_mean'] for r in cfg_results])*100:.2f}%")
    
    # Save all results
    with open(os.path.join(output, "pareto_results.json"), 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2, default=str)
    
    # Summary table
    logger.info("\n" + "=" * 75)
    logger.info("PARETO SUMMARY")
    logger.info("=" * 75)
    logger.info(f"{'Config':<18} {'Cost':<10} {'FillRate':<10} {'ServiceLv':<10} {'Bullwhip':<10}")
    logger.info("-" * 75)
    
    for wc in weights:
        name = wc.get("name", f"cw{wc['cost_weight']}_sw{wc['service_weight']}")
        cr = [r for r in results if r.get("config_name") == name]
        if cr:
            logger.info(f"{name:<18} "
                       f"{np.mean([r['total_cost'] for r in cr]):<10.1f} "
                       f"{np.mean([r['fill_rate_mean'] for r in cr])*100:<10.2f}% "
                       f"{np.mean([r['service_level_mean'] for r in cr])*100:<10.2f}% "
                       f"{np.mean([r['bullwhip_mean'] for r in cr]):<10.3f}")
    
    logger.info("=" * 75)
    logger.info(f"Results saved to: {output}/pareto_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pareto_network.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
