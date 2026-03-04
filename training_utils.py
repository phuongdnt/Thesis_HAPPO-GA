"""
training_utils.py
==================
Professional training utilities with advanced early stopping mechanisms.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import deque


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    enabled: bool = True
    patience: int = 15                    # Number of evaluations without improvement
    min_delta: float = 0.01               # Minimum improvement to count as progress (1%)
    warmup_evaluations: int = 10          # Don't stop during warmup phase
    
    # Multi-metric tracking
    primary_metric: str = "score"         # Main metric to track: "score", "cost", "fill_rate"
    secondary_metrics: List[str] = field(default_factory=lambda: ["cost", "fill_rate", "bullwhip"])
    
    # Trend detection
    trend_window: int = 5                 # Window for detecting deteriorating trend
    max_deterioration_streak: int = 3     # Stop if metrics worsen this many times in a row
    
    # LR reduction
    reduce_lr_on_plateau: bool = True
    lr_reduction_factor: float = 0.5      # Multiply LR by this when plateau detected
    lr_reduction_patience: int = 5        # Reduce LR after this many evals without improvement
    min_lr: float = 1e-6                  # Don't reduce below this


class EarlyStoppingTracker:
    """
    Professional early stopping tracker with multiple stopping criteria.
    
    Stopping conditions:
    1. No improvement in primary metric for `patience` evaluations
    2. Sustained deterioration trend detected
    3. All metrics are worse than best for `max_deterioration_streak` times
    """
    
    def __init__(self, config: EarlyStoppingConfig, logger=None):
        self.config = config
        self.logger = logger
        
        # Best tracking
        self.best_score: Optional[float] = None
        self.best_metrics: Dict[str, float] = {}
        self.best_episode: int = 0
        self.best_model_state: Optional[Dict] = None
        
        # Patience counters
        self.no_improve_count: int = 0
        self.lr_no_improve_count: int = 0
        self.deterioration_streak: int = 0
        
        # History for trend detection
        self.score_history: deque = deque(maxlen=config.trend_window * 2)
        self.metrics_history: List[Dict] = []
        
        # Evaluation counter
        self.eval_count: int = 0
        
        # LR reduction tracking
        self.lr_reductions: int = 0
        self.current_lr_factor: float = 1.0
        
    def update(self, episode: int, metrics: Dict[str, float], 
               model_state: Optional[Dict] = None) -> Tuple[bool, str, Optional[float]]:
        """
        Update tracker with new evaluation metrics.
        
        Args:
            episode: Current episode number
            metrics: Dictionary with evaluation metrics (must include keys for tracking)
            model_state: Optional model state to save if this is best
            
        Returns:
            Tuple of (should_stop, reason, suggested_lr_factor)
            - should_stop: True if training should stop
            - reason: Explanation string
            - suggested_lr_factor: New LR multiplier if LR should be reduced, None otherwise
        """
        self.eval_count += 1
        
        # Compute combined score (negative cost + positive service)
        # This should match how you compute eval_score in training
        score = metrics.get("score", self._compute_score(metrics))
        
        self.score_history.append(score)
        self.metrics_history.append(metrics.copy())
        
        # Check if in warmup phase
        if self.eval_count <= self.config.warmup_evaluations:
            if self.best_score is None or score > self.best_score:
                self._update_best(episode, score, metrics, model_state)
            return False, "warmup", None
        
        # Check for improvement
        improved = self._check_improvement(score)
        
        if improved:
            self._update_best(episode, score, metrics, model_state)
            self.no_improve_count = 0
            self.lr_no_improve_count = 0
            self.deterioration_streak = 0
            return False, "improved", None
        else:
            self.no_improve_count += 1
            self.lr_no_improve_count += 1
            
            # Check deterioration
            if self._is_deteriorating(metrics):
                self.deterioration_streak += 1
            else:
                self.deterioration_streak = 0
        
        # Check stopping conditions
        should_stop, reason = self._check_stopping_conditions()
        
        # Check LR reduction
        lr_factor = None
        if not should_stop and self.config.reduce_lr_on_plateau:
            lr_factor = self._check_lr_reduction()
        
        return should_stop, reason, lr_factor
    
    def _compute_score(self, metrics: Dict) -> float:
        """Compute combined score from metrics."""
        # Default: negative cost + service bonus
        cost = metrics.get("total_cost", 0)
        fill_rate = metrics.get("fill_rate_mean", 0)
        return -cost + fill_rate * 1000
    
    def _check_improvement(self, score: float) -> bool:
        """Check if score improved by at least min_delta."""
        if self.best_score is None:
            return True
        
        # Relative improvement
        if self.best_score != 0:
            rel_improvement = (score - self.best_score) / abs(self.best_score)
            return rel_improvement > self.config.min_delta
        else:
            return score > self.best_score + self.config.min_delta
    
    def _is_deteriorating(self, metrics: Dict) -> bool:
        """Check if all tracked metrics are worse than best."""
        if not self.best_metrics:
            return False
        
        worse_count = 0
        total_count = 0
        
        # Cost should decrease
        if "total_cost" in metrics and "total_cost" in self.best_metrics:
            total_count += 1
            if metrics["total_cost"] > self.best_metrics["total_cost"] * 1.05:  # 5% worse
                worse_count += 1
        
        # Fill rate should increase
        if "fill_rate_mean" in metrics and "fill_rate_mean" in self.best_metrics:
            total_count += 1
            if metrics["fill_rate_mean"] < self.best_metrics["fill_rate_mean"] * 0.95:
                worse_count += 1
        
        # Bullwhip should decrease
        if "bullwhip_mean" in metrics and "bullwhip_mean" in self.best_metrics:
            total_count += 1
            if metrics["bullwhip_mean"] > self.best_metrics["bullwhip_mean"] * 1.1:
                worse_count += 1
        
        return worse_count >= total_count and total_count > 0
    
    def _check_stopping_conditions(self) -> Tuple[bool, str]:
        """Check all stopping conditions."""
        
        # Condition 1: Patience exceeded
        if self.no_improve_count >= self.config.patience:
            return True, f"no_improvement_for_{self.no_improve_count}_evals"
        
        # Condition 2: Sustained deterioration
        if self.deterioration_streak >= self.config.max_deterioration_streak:
            return True, f"deterioration_streak_{self.deterioration_streak}"
        
        # Condition 3: Trend analysis (if enough history)
        if len(self.score_history) >= self.config.trend_window * 2:
            recent = list(self.score_history)[-self.config.trend_window:]
            older = list(self.score_history)[-self.config.trend_window*2:-self.config.trend_window]
            
            recent_mean = np.mean(recent)
            older_mean = np.mean(older)
            
            # If recent scores are significantly worse than older ones
            if older_mean > 0 and (older_mean - recent_mean) / abs(older_mean) > 0.1:
                # AND no improvement for a while
                if self.no_improve_count >= self.config.patience // 2:
                    return True, "negative_trend_detected"
        
        return False, "continue"
    
    def _check_lr_reduction(self) -> Optional[float]:
        """Check if LR should be reduced."""
        if self.lr_no_improve_count >= self.config.lr_reduction_patience:
            new_factor = self.current_lr_factor * self.config.lr_reduction_factor
            
            if new_factor >= self.config.min_lr / 0.0001:  # Assuming base LR ~ 0.0001
                self.current_lr_factor = new_factor
                self.lr_no_improve_count = 0
                self.lr_reductions += 1
                
                if self.logger:
                    self.logger.info(f"  [LR] Reducing learning rate by {self.config.lr_reduction_factor}x "
                                    f"(reduction #{self.lr_reductions})")
                
                return new_factor
        
        return None
    
    def _update_best(self, episode: int, score: float, 
                     metrics: Dict, model_state: Optional[Dict]):
        """Update best tracking."""
        self.best_score = score
        self.best_metrics = metrics.copy()
        self.best_episode = episode
        if model_state is not None:
            self.best_model_state = model_state
    
    def get_summary(self) -> Dict:
        """Get summary of training progress."""
        return {
            "best_score": self.best_score,
            "best_episode": self.best_episode,
            "best_metrics": self.best_metrics,
            "total_evaluations": self.eval_count,
            "no_improve_streak": self.no_improve_count,
            "lr_reductions": self.lr_reductions,
            "final_lr_factor": self.current_lr_factor,
        }


class RewardSmoother:
    """Exponential moving average smoother for reward tracking."""
    
    def __init__(self, alpha: float = 0.1, window: int = 50):
        self.alpha = alpha
        self.window = window
        self.ema: Optional[float] = None
        self.history: deque = deque(maxlen=window)
        
    def update(self, reward: float) -> float:
        """Update and return smoothed reward."""
        self.history.append(reward)
        
        if self.ema is None:
            self.ema = reward
        else:
            self.ema = self.alpha * reward + (1 - self.alpha) * self.ema
        
        return self.ema
    
    def get_stats(self) -> Dict[str, float]:
        """Get reward statistics."""
        if len(self.history) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "ema": 0}
        
        arr = np.array(self.history)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "ema": float(self.ema) if self.ema else 0,
        }


class TrainingProgressMonitor:
    """Monitor overall training progress and detect issues."""
    
    def __init__(self, check_interval: int = 100):
        self.check_interval = check_interval
        self.episode_rewards: List[float] = []
        self.eval_metrics: List[Dict] = []
        self.warnings: List[str] = []
        
    def log_episode(self, episode: int, reward: float):
        """Log episode reward."""
        self.episode_rewards.append(reward)
        
        # Periodic health check
        if episode % self.check_interval == 0 and len(self.episode_rewards) >= self.check_interval:
            self._health_check(episode)
    
    def log_evaluation(self, metrics: Dict):
        """Log evaluation metrics."""
        self.eval_metrics.append(metrics)
    
    def _health_check(self, episode: int) -> List[str]:
        """Run health checks and return warnings."""
        warnings = []
        recent = self.episode_rewards[-self.check_interval:]
        
        # Check for NaN/Inf
        if any(not np.isfinite(r) for r in recent):
            warnings.append(f"[Episode {episode}] WARNING: NaN/Inf rewards detected!")
        
        # Check for reward collapse
        if np.std(recent) < 1e-6:
            warnings.append(f"[Episode {episode}] WARNING: Reward variance collapsed - possible dead policy")
        
        # Check for extreme values
        if np.max(np.abs(recent)) > 1e6:
            warnings.append(f"[Episode {episode}] WARNING: Extreme reward values detected")
        
        # Check for improvement (compare first and second half)
        if len(self.episode_rewards) >= self.check_interval * 2:
            older = self.episode_rewards[-self.check_interval*2:-self.check_interval]
            if np.mean(recent) < np.mean(older) * 0.8:
                warnings.append(f"[Episode {episode}] WARNING: Recent rewards 20%+ worse than before")
        
        self.warnings.extend(warnings)
        return warnings
    
    def get_report(self) -> str:
        """Generate training progress report."""
        if len(self.episode_rewards) == 0:
            return "No training data yet."
        
        report = []
        report.append("=" * 60)
        report.append("TRAINING PROGRESS REPORT")
        report.append("=" * 60)
        
        # Overall stats
        arr = np.array(self.episode_rewards)
        report.append(f"Total episodes: {len(arr)}")
        report.append(f"Reward - Mean: {np.mean(arr):.2f}, Std: {np.std(arr):.2f}")
        report.append(f"Reward - Min: {np.min(arr):.2f}, Max: {np.max(arr):.2f}")
        
        # Recent vs early comparison
        if len(arr) >= 200:
            early = arr[:100]
            recent = arr[-100:]
            change = (np.mean(recent) - np.mean(early)) / abs(np.mean(early)) * 100 if np.mean(early) != 0 else 0
            report.append(f"Improvement (last 100 vs first 100): {change:+.1f}%")
        
        # Warnings
        if self.warnings:
            report.append("\nWarnings:")
            for w in self.warnings[-5:]:  # Last 5 warnings
                report.append(f"  {w}")
        
        return "\n".join(report)


def apply_lr_factor(agent, factor: float):
    """Apply learning rate factor to all optimizers in agent."""
    for optimizer in agent.actor_optimisers:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor
    
    for param_group in agent.critic_optimizer.param_groups:
        param_group['lr'] *= factor
