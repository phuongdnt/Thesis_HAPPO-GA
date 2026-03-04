"""
evaluate_all.py - Comprehensive Evaluation for Thesis
======================================================
Evaluates:
1. HAPPO + GA Hybrid (all Pareto configs)
2. Pure HAPPO (if available)
3. Base-stock policy (classical baseline)
4. (s,S) policy (classical baseline)

Outputs:
- Comparison tables (LaTeX + CSV)
- Statistical significance tests
- Summary for thesis Chapter 5

Usage: python evaluate_all.py --serial results/pareto_serial --network results/pareto_network
"""

import os
import json
import argparse
from typing import Dict, List
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Statistical tests will be limited.")

from envs.serial_env import SerialInventoryEnv
from baselines import (
    BaseStockPolicy, sSPolicy,
    compute_optimal_base_stock_levels,
    compute_optimal_sS_params,
    evaluate_baseline_policy
)


def load_pareto_results(results_dir: str) -> List[Dict]:
    """Load Pareto results from JSON."""
    results_file = os.path.join(results_dir, "pareto_results.json")
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} not found")
        return []
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data.get("results", [])


def evaluate_baselines_serial(
    backlog_cost: List[float] = [5.0, 3.0, 2.0],
    holding_cost: List[float] = [1.0, 1.0, 1.0],
    mean_demand: float = 10.0,
    std_demand: float = 3.0,
    n_episodes: int = 30,
) -> Dict[str, Dict]:
    """Evaluate classical baselines on serial topology."""
    
    print("\n" + "=" * 60)
    print("EVALUATING CLASSICAL BASELINES - SERIAL")
    print("=" * 60)
    
    # Build environment
    env = SerialInventoryEnv(
        lead_time=2,
        episode_len=100,
        action_dim=41,
        init_inventory=20,
        init_outstanding=10,
        holding_cost=holding_cost,
        backlog_cost=backlog_cost,
        fixed_cost=1.0,
    )
    
    results = {}
    
    # 1. Base-stock policy
    print("\n[1] Base-stock Policy")
    base_stock_levels = compute_optimal_base_stock_levels(
        mean_demand, std_demand, 
        lead_time=2,
        holding_cost=holding_cost,
        backlog_cost=backlog_cost,
    )
    print(f"    Target levels: {[f'{s:.1f}' for s in base_stock_levels]}")
    
    base_stock_policy = BaseStockPolicy(target_levels=base_stock_levels)
    results["Base-stock"] = evaluate_baseline_policy(
        env, base_stock_policy, n_episodes, "Base-stock"
    )
    
    # 2. (s,S) policy
    print("\n[2] (s,S) Policy")
    s_points, S_levels = compute_optimal_sS_params(
        mean_demand, std_demand,
        lead_time=2,
        holding_cost=holding_cost,
        backlog_cost=backlog_cost,
    )
    print(f"    Reorder points (s): {[f'{s:.1f}' for s in s_points]}")
    print(f"    Target levels (S): {[f'{S:.1f}' for S in S_levels]}")
    
    sS_policy = sSPolicy(reorder_points=s_points, target_levels=S_levels)
    results["(s,S)"] = evaluate_baseline_policy(
        env, sS_policy, n_episodes, "(s,S)"
    )
    
    return results


def aggregate_pareto_by_config(results: List[Dict]) -> Dict[str, Dict]:
    """Aggregate Pareto results by config name (mean ± std across seeds)."""
    configs = {}
    
    for r in results:
        name = r.get("config_name", "unknown")
        if name not in configs:
            configs[name] = []
        configs[name].append(r)
    
    aggregated = {}
    for name, runs in configs.items():
        aggregated[name] = {
            "total_cost": np.mean([r["total_cost"] for r in runs]),
            "total_cost_std": np.std([r["total_cost"] for r in runs]),
            "fill_rate_mean": np.mean([r["fill_rate_mean"] for r in runs]),
            "fill_rate_std": np.std([r["fill_rate_mean"] for r in runs]),
            "service_level_mean": np.mean([r["service_level_mean"] for r in runs]),
            "service_level_std": np.std([r["service_level_mean"] for r in runs]),
            "bullwhip_mean": np.mean([r["bullwhip_mean"] for r in runs]),
            "bullwhip_std": np.std([r["bullwhip_mean"] for r in runs]),
            "n_runs": len(runs),
            "cost_weight": runs[0].get("cost_weight", 0.5),
            "service_weight": runs[0].get("service_weight", 0.5),
        }
    
    return aggregated


def run_statistical_tests(hybrid_results: List[Dict], baseline_results: Dict) -> Dict:
    """Run statistical tests comparing Hybrid vs Baselines."""
    
    if not HAS_SCIPY:
        print("Warning: scipy not installed, skipping statistical tests")
        return {}
    
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)
    
    tests = {}
    
    # Get balanced config results
    balanced_runs = [r for r in hybrid_results if r.get("config_name") == "balanced"]
    if not balanced_runs:
        balanced_runs = hybrid_results[:3] if len(hybrid_results) >= 3 else hybrid_results
    
    hybrid_costs = [r["total_cost"] for r in balanced_runs]
    hybrid_fr = [r["fill_rate_mean"] for r in balanced_runs]
    hybrid_bw = [r["bullwhip_mean"] for r in balanced_runs]
    
    # Compare with each baseline
    for baseline_name, baseline_metrics in baseline_results.items():
        print(f"\n[Hybrid vs {baseline_name}]")
        
        # For baselines, we only have single values, so use one-sample t-test
        baseline_cost = baseline_metrics["total_cost"]
        baseline_fr = baseline_metrics["fill_rate_mean"]
        baseline_bw = baseline_metrics["bullwhip_mean"]
        
        # Cost comparison (lower is better)
        t_cost, p_cost = stats.ttest_1samp(hybrid_costs, baseline_cost)
        cost_better = np.mean(hybrid_costs) < baseline_cost
        print(f"  Cost: Hybrid={np.mean(hybrid_costs):.1f} vs {baseline_name}={baseline_cost:.1f}")
        print(f"        t={t_cost:.3f}, p={p_cost:.4f}, Hybrid {'better' if cost_better else 'worse'}")
        
        # Fill rate comparison (higher is better)
        t_fr, p_fr = stats.ttest_1samp(hybrid_fr, baseline_fr)
        fr_better = np.mean(hybrid_fr) > baseline_fr
        print(f"  FR:   Hybrid={np.mean(hybrid_fr)*100:.1f}% vs {baseline_name}={baseline_fr*100:.1f}%")
        print(f"        t={t_fr:.3f}, p={p_fr:.4f}, Hybrid {'better' if fr_better else 'worse'}")
        
        # Bullwhip comparison (lower is better)
        t_bw, p_bw = stats.ttest_1samp(hybrid_bw, baseline_bw)
        bw_better = np.mean(hybrid_bw) < baseline_bw
        print(f"  BW:   Hybrid={np.mean(hybrid_bw):.3f} vs {baseline_name}={baseline_bw:.3f}")
        print(f"        t={t_bw:.3f}, p={p_bw:.4f}, Hybrid {'better' if bw_better else 'worse'}")
        
        tests[baseline_name] = {
            "cost": {"t": t_cost, "p": p_cost, "hybrid_better": cost_better},
            "fill_rate": {"t": t_fr, "p": p_fr, "hybrid_better": fr_better},
            "bullwhip": {"t": t_bw, "p": p_bw, "hybrid_better": bw_better},
        }
    
    return tests


def print_comparison_table(
    pareto_agg: Dict[str, Dict], 
    baseline_results: Dict[str, Dict],
    topology: str = "Serial"
):
    """Print comparison table for thesis."""
    
    print("\n" + "=" * 80)
    print(f"COMPARISON TABLE - {topology.upper()} TOPOLOGY")
    print("=" * 80)
    
    # Header
    print(f"\n{'Method':<20} {'Cost':<15} {'Fill Rate':<12} {'Service Lv':<12} {'Bullwhip':<10}")
    print("-" * 80)
    
    # Baselines
    print("Classical Baselines:")
    for name, metrics in baseline_results.items():
        print(f"  {name:<18} {metrics['total_cost']:<15.1f} "
              f"{metrics['fill_rate_mean']*100:<12.1f}% "
              f"{metrics['service_level_mean']*100:<12.1f}% "
              f"{metrics['bullwhip_mean']:<10.3f}")
    
    # HAPPO + GA Hybrid
    print("\nHAPPO + GA Hybrid (Pareto):")
    for name in ["cost_focused", "cost_leaning", "balanced", "service_leaning", "service_focused"]:
        if name in pareto_agg:
            m = pareto_agg[name]
            print(f"  {name:<18} {m['total_cost']:<15.1f} "
                  f"{m['fill_rate_mean']*100:<12.1f}% "
                  f"{m['service_level_mean']*100:<12.1f}% "
                  f"{m['bullwhip_mean']:<10.3f}")
    
    print("-" * 80)


def generate_latex_table(
    pareto_agg: Dict[str, Dict],
    baseline_results: Dict[str, Dict],
    output_file: str
):
    """Generate LaTeX table for thesis."""
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Performance Comparison: HAPPO+GA Hybrid vs Classical Baselines}",
        "\\label{tab:comparison}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Total Cost & Fill Rate (\\%) & Service Level (\\%) & Bullwhip \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\textit{Classical Baselines}} \\\\",
    ]
    
    # Baselines
    for name, m in baseline_results.items():
        lines.append(f"{name} & {m['total_cost']:.1f} & {m['fill_rate_mean']*100:.1f} & "
                    f"{m['service_level_mean']*100:.1f} & {m['bullwhip_mean']:.3f} \\\\")
    
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\textit{HAPPO + GA Hybrid (Proposed)}} \\\\")
    
    # Pareto configs
    for name in ["cost_focused", "cost_leaning", "balanced", "service_leaning", "service_focused"]:
        if name in pareto_agg:
            m = pareto_agg[name]
            display_name = name.replace("_", " ").title()
            lines.append(f"{display_name} & {m['total_cost']:.1f} & {m['fill_rate_mean']*100:.1f} & "
                        f"{m['service_level_mean']*100:.1f} & {m['bullwhip_mean']:.3f} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nLaTeX table saved to: {output_file}")


def generate_csv_results(
    pareto_agg: Dict[str, Dict],
    baseline_results: Dict[str, Dict],
    output_file: str
):
    """Generate CSV for further analysis."""
    
    lines = ["method,type,cost,cost_std,fill_rate,fill_rate_std,service_level,service_level_std,bullwhip,bullwhip_std"]
    
    # Baselines
    for name, m in baseline_results.items():
        lines.append(f"{name},baseline,{m['total_cost']:.2f},0,"
                    f"{m['fill_rate_mean']:.4f},0,"
                    f"{m['service_level_mean']:.4f},0,"
                    f"{m['bullwhip_mean']:.4f},0")
    
    # Pareto
    for name, m in pareto_agg.items():
        lines.append(f"{name},hybrid,{m['total_cost']:.2f},{m['total_cost_std']:.2f},"
                    f"{m['fill_rate_mean']:.4f},{m['fill_rate_std']:.4f},"
                    f"{m['service_level_mean']:.4f},{m['service_level_std']:.4f},"
                    f"{m['bullwhip_mean']:.4f},{m['bullwhip_std']:.4f}")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"CSV results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="results/pareto_serial", help="Serial results dir")
    parser.add_argument("--network", default="results/pareto_network", help="Network results dir")
    parser.add_argument("--output", default="results/analysis", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("COMPREHENSIVE EVALUATION FOR THESIS")
    print("=" * 60)
    
    # === SERIAL TOPOLOGY ===
    print("\n" + "=" * 60)
    print("SERIAL TOPOLOGY")
    print("=" * 60)
    
    # Load Pareto results
    serial_results = load_pareto_results(args.serial)
    if serial_results:
        serial_agg = aggregate_pareto_by_config(serial_results)
        print(f"Loaded {len(serial_results)} Pareto runs")
    else:
        serial_agg = {}
        print("No Pareto results found for serial")
    
    # Evaluate baselines
    baseline_serial = evaluate_baselines_serial()
    
    # Print comparison
    if serial_agg:
        print_comparison_table(serial_agg, baseline_serial, "Serial")
        
        # Statistical tests
        stat_tests = run_statistical_tests(serial_results, baseline_serial)
        
        # Generate outputs
        generate_latex_table(serial_agg, baseline_serial, 
                           os.path.join(args.output, "table_serial.tex"))
        generate_csv_results(serial_agg, baseline_serial,
                           os.path.join(args.output, "results_serial.csv"))
    
    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("SUMMARY FOR THESIS")
    print("=" * 60)
    
    if serial_agg and "balanced" in serial_agg:
        hybrid = serial_agg["balanced"]
        base_stock = baseline_serial.get("Base-stock", {})
        sS = baseline_serial.get("(s,S)", {})
        
        print("\nKey Findings (Balanced Config vs Baselines):")
        
        if base_stock:
            cost_improve = (base_stock["total_cost"] - hybrid["total_cost"]) / base_stock["total_cost"] * 100
            fr_improve = (hybrid["fill_rate_mean"] - base_stock["fill_rate_mean"]) * 100
            bw_improve = (base_stock["bullwhip_mean"] - hybrid["bullwhip_mean"]) / base_stock["bullwhip_mean"] * 100
            
            print(f"\n  vs Base-stock:")
            print(f"    Cost: {cost_improve:+.1f}% {'(better)' if cost_improve > 0 else '(worse)'}")
            print(f"    Fill Rate: {fr_improve:+.1f}pp {'(better)' if fr_improve > 0 else '(worse)'}")
            print(f"    Bullwhip: {bw_improve:+.1f}% {'(better)' if bw_improve > 0 else '(worse)'}")
        
        if sS:
            cost_improve = (sS["total_cost"] - hybrid["total_cost"]) / sS["total_cost"] * 100
            fr_improve = (hybrid["fill_rate_mean"] - sS["fill_rate_mean"]) * 100
            bw_improve = (sS["bullwhip_mean"] - hybrid["bullwhip_mean"]) / sS["bullwhip_mean"] * 100
            
            print(f"\n  vs (s,S):")
            print(f"    Cost: {cost_improve:+.1f}% {'(better)' if cost_improve > 0 else '(worse)'}")
            print(f"    Fill Rate: {fr_improve:+.1f}pp {'(better)' if fr_improve > 0 else '(worse)'}")
            print(f"    Bullwhip: {bw_improve:+.1f}% {'(better)' if bw_improve > 0 else '(worse)'}")
    
    print("\n" + "=" * 60)
    print(f"Analysis outputs saved to: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
