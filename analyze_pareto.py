"""
analyze_pareto.py - Post-Training Analysis for Thesis
======================================================

This script analyzes results AFTER training is complete.
It generates:
1. Pareto frontier plots (Cost vs Fill Rate)
2. Per-agent metrics comparison tables
3. Bullwhip effect analysis across echelons
4. Statistical significance tests
5. Comparison with baseline
6. Publication-ready figures and tables

Usage:
    python analyze_pareto.py --serial results/pareto_serial --network results/pareto_network

Author: Meo
Thesis: Multi-Agent Deep Reinforcement Learning for Multi-Echelon
        Inventory Systems Under Demand Uncertainty
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be skipped.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Using basic tables.")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Statistical tests will be skipped.")


@dataclass
class ExperimentResult:
    """Container for a single experiment result."""
    config_name: str
    cost_weight: float
    service_weight: float
    seed: int
    total_cost: float
    fill_rate: float
    bullwhip: float
    cycle_sl: float
    cost_per_agent: List[float]
    fill_rate_per_agent: List[float]
    bullwhip_per_agent: List[float]
    cycle_sl_per_agent: List[float]
    training_time: float
    episodes_trained: int
    stop_reason: str


def load_results(results_dir: str) -> List[ExperimentResult]:
    """Load all results from a pareto results directory."""
    results_file = os.path.join(results_dir, "pareto_results.json")
    
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} not found")
        return []
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = []
    for r in data.get("results", []):
        try:
            results.append(ExperimentResult(
                config_name=r.get("config_name", "unknown"),
                cost_weight=r.get("cost_weight", 0),
                service_weight=r.get("service_weight", 0),
                seed=r.get("seed", 0),
                total_cost=r.get("total_cost", 0),
                fill_rate=r.get("fill_rate_mean", 0),
                bullwhip=r.get("bullwhip_mean", 0),
                cycle_sl=r.get("cycle_sl_mean", 0),
                cost_per_agent=r.get("cost_per_agent", []),
                fill_rate_per_agent=r.get("fill_rate_per_agent", []),
                bullwhip_per_agent=r.get("bullwhip_per_agent", []),
                cycle_sl_per_agent=r.get("cycle_sl_per_agent", []),
                training_time=r.get("training_time", 0),
                episodes_trained=r.get("episodes_trained", 0),
                stop_reason=r.get("stop_reason", "unknown"),
            ))
        except Exception as e:
            print(f"Warning: Failed to parse result: {e}")
    
    return results


def aggregate_by_config(results: List[ExperimentResult]) -> Dict[str, Dict]:
    """Aggregate results by config name (across seeds)."""
    configs = {}
    
    for r in results:
        if r.config_name not in configs:
            configs[r.config_name] = {
                "cost_weight": r.cost_weight,
                "service_weight": r.service_weight,
                "results": []
            }
        configs[r.config_name]["results"].append(r)
    
    # Compute statistics
    for config_name, config_data in configs.items():
        rs = config_data["results"]
        n = len(rs)
        
        config_data["n_seeds"] = n
        config_data["cost_mean"] = np.mean([r.total_cost for r in rs])
        config_data["cost_std"] = np.std([r.total_cost for r in rs])
        config_data["fill_rate_mean"] = np.mean([r.fill_rate for r in rs])
        config_data["fill_rate_std"] = np.std([r.fill_rate for r in rs])
        config_data["bullwhip_mean"] = np.mean([r.bullwhip for r in rs])
        config_data["bullwhip_std"] = np.std([r.bullwhip for r in rs])
        config_data["cycle_sl_mean"] = np.mean([r.cycle_sl for r in rs])
        config_data["cycle_sl_std"] = np.std([r.cycle_sl for r in rs])
        
        # Per-agent stats
        if rs[0].cost_per_agent:
            n_agents = len(rs[0].cost_per_agent)
            config_data["cost_per_agent_mean"] = [
                np.mean([r.cost_per_agent[i] for r in rs]) for i in range(n_agents)
            ]
            config_data["fill_rate_per_agent_mean"] = [
                np.mean([r.fill_rate_per_agent[i] for r in rs]) for i in range(n_agents)
            ]
            config_data["bullwhip_per_agent_mean"] = [
                np.mean([r.bullwhip_per_agent[i] for r in rs]) for i in range(n_agents)
            ]
    
    return configs


def print_summary_table(configs: Dict[str, Dict], topology: str):
    """Print summary table for thesis."""
    print(f"\n{'='*80}")
    print(f"PARETO FRONTIER SUMMARY - {topology.upper()}")
    print(f"{'='*80}")
    
    # Sort by cost_weight descending
    sorted_configs = sorted(configs.items(), key=lambda x: x[1]["cost_weight"], reverse=True)
    
    print(f"\n{'Config':<18} {'α':<5} {'β':<5} {'Cost':<16} {'Fill Rate':<16} {'Bullwhip':<14}")
    print("-" * 80)
    
    for config_name, data in sorted_configs:
        cost_str = f"{data['cost_mean']:.1f} ± {data['cost_std']:.1f}"
        fr_str = f"{data['fill_rate_mean']*100:.2f}% ± {data['fill_rate_std']*100:.2f}%"
        bw_str = f"{data['bullwhip_mean']:.3f} ± {data['bullwhip_std']:.3f}"
        
        print(f"{config_name:<18} {data['cost_weight']:<5.1f} {data['service_weight']:<5.1f} "
              f"{cost_str:<16} {fr_str:<16} {bw_str:<14}")
    
    print("-" * 80)


def print_per_agent_table(configs: Dict[str, Dict], topology: str, agent_names: List[str]):
    """Print per-agent breakdown table."""
    print(f"\n{'='*80}")
    print(f"PER-AGENT METRICS - {topology.upper()}")
    print(f"{'='*80}")
    
    sorted_configs = sorted(configs.items(), key=lambda x: x[1]["cost_weight"], reverse=True)
    
    for config_name, data in sorted_configs:
        print(f"\n>>> {config_name} (α={data['cost_weight']}, β={data['service_weight']})")
        print(f"{'Agent':<15} {'Cost':<12} {'Fill Rate':<12} {'Bullwhip':<12}")
        print("-" * 55)
        
        if "cost_per_agent_mean" in data:
            n_agents = len(data["cost_per_agent_mean"])
            for i in range(n_agents):
                name = agent_names[i] if i < len(agent_names) else f"Agent_{i}"
                print(f"{name:<15} {data['cost_per_agent_mean'][i]:<12.1f} "
                      f"{data['fill_rate_per_agent_mean'][i]*100:<12.2f}% "
                      f"{data['bullwhip_per_agent_mean'][i]:<12.3f}")
        
        print("-" * 55)
        print(f"{'TOTAL':<15} {data['cost_mean']:<12.1f} "
              f"{data['fill_rate_mean']*100:<12.2f}% "
              f"{data['bullwhip_mean']:<12.3f}")


def analyze_bullwhip_amplification(configs: Dict[str, Dict], agent_names: List[str]):
    """Analyze bullwhip effect amplification across echelons."""
    print(f"\n{'='*80}")
    print("BULLWHIP EFFECT ANALYSIS")
    print("(Ratio of upstream to downstream bullwhip)")
    print(f"{'='*80}")
    
    sorted_configs = sorted(configs.items(), key=lambda x: x[1]["cost_weight"], reverse=True)
    
    for config_name, data in sorted_configs:
        if "bullwhip_per_agent_mean" not in data:
            continue
        
        bw = data["bullwhip_per_agent_mean"]
        n_agents = len(bw)
        
        print(f"\n>>> {config_name}")
        
        # For serial: Retailer -> Distributor -> Factory
        # For network: Retailers -> Distributors -> Factories
        if n_agents == 3:  # Serial
            print(f"  Retailer:    {bw[0]:.3f}")
            print(f"  Distributor: {bw[1]:.3f} (amplification: {bw[1]/bw[0]:.2f}x vs Retailer)")
            print(f"  Factory:     {bw[2]:.3f} (amplification: {bw[2]/bw[0]:.2f}x vs Retailer)")
            
            # Check if bullwhip is controlled
            if bw[2] < bw[0]:
                print(f"  ✅ Bullwhip REDUCED at upstream (good!)")
            else:
                print(f"  ⚠️ Bullwhip AMPLIFIED at upstream")
                
        elif n_agents == 6:  # Network
            avg_retailer = np.mean(bw[0:2])
            avg_distributor = np.mean(bw[2:4])
            avg_factory = np.mean(bw[4:6])
            
            print(f"  Retailers (avg):    {avg_retailer:.3f}")
            print(f"  Distributors (avg): {avg_distributor:.3f} (amplification: {avg_distributor/avg_retailer:.2f}x)")
            print(f"  Factories (avg):    {avg_factory:.3f} (amplification: {avg_factory/avg_retailer:.2f}x)")


def compute_pareto_frontier(configs: Dict[str, Dict]) -> List[str]:
    """Identify Pareto-optimal configurations."""
    points = []
    for config_name, data in configs.items():
        # Minimize cost, maximize fill_rate
        points.append({
            "name": config_name,
            "cost": data["cost_mean"],
            "fill_rate": data["fill_rate_mean"]
        })
    
    # Find Pareto-optimal points
    pareto_optimal = []
    for p in points:
        is_dominated = False
        for q in points:
            # q dominates p if q is better in both objectives
            if q["cost"] < p["cost"] and q["fill_rate"] > p["fill_rate"]:
                is_dominated = True
                break
        if not is_dominated:
            pareto_optimal.append(p["name"])
    
    return pareto_optimal


def plot_pareto_frontier(configs: Dict[str, Dict], topology: str, output_dir: str):
    """Plot Pareto frontier (Cost vs Fill Rate)."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot (matplotlib not installed)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color map for different configs
    colors = {
        "cost_focused": "#e74c3c",      # Red
        "cost_leaning": "#e67e22",      # Orange
        "balanced": "#27ae60",          # Green
        "service_leaning": "#3498db",   # Blue
        "service_focused": "#9b59b6",   # Purple
    }
    
    pareto_points = compute_pareto_frontier(configs)
    
    # Plot each config
    for config_name, data in configs.items():
        color = colors.get(config_name, "#7f8c8d")
        is_pareto = config_name in pareto_points
        
        # Plot mean with error bars
        ax.errorbar(
            data["cost_mean"], 
            data["fill_rate_mean"] * 100,
            xerr=data["cost_std"],
            yerr=data["fill_rate_std"] * 100,
            fmt='o' if is_pareto else 's',
            color=color,
            markersize=12 if is_pareto else 8,
            markeredgecolor='black' if is_pareto else color,
            markeredgewidth=2 if is_pareto else 1,
            capsize=5,
            label=f"{config_name} (α={data['cost_weight']}, β={data['service_weight']})"
        )
    
    # Connect Pareto-optimal points
    pareto_data = [(configs[name]["cost_mean"], configs[name]["fill_rate_mean"] * 100) 
                   for name in pareto_points]
    pareto_data.sort(key=lambda x: x[0])
    if len(pareto_data) > 1:
        xs, ys = zip(*pareto_data)
        ax.plot(xs, ys, 'k--', alpha=0.5, linewidth=1.5, label='Pareto Frontier')
    
    ax.set_xlabel('Total Cost', fontsize=12)
    ax.set_ylabel('Fill Rate (%)', fontsize=12)
    ax.set_title(f'Pareto Frontier - {topology.upper()} Topology\n'
                 f'(Multi-Objective Trade-off: Cost vs Service Level)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for direction
    ax.annotate('', xy=(ax.get_xlim()[0], ax.get_ylim()[1]), 
                xytext=(ax.get_xlim()[0] + (ax.get_xlim()[1]-ax.get_xlim()[0])*0.1, 
                        ax.get_ylim()[1] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.1),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1]-ax.get_xlim()[0])*0.05, 
            ax.get_ylim()[1] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05,
            'Ideal\nDirection', fontsize=10, color='green')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f"pareto_frontier_{topology}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    output_path_pdf = os.path.join(output_dir, f"pareto_frontier_{topology}.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")
    
    plt.close()


def plot_bullwhip_comparison(configs: Dict[str, Dict], topology: str, 
                              agent_names: List[str], output_dir: str):
    """Plot bullwhip effect comparison across configs and echelons."""
    if not HAS_MATPLOTLIB:
        return
    
    sorted_configs = sorted(configs.items(), key=lambda x: x[1]["cost_weight"], reverse=True)
    config_names = [c[0] for c in sorted_configs]
    
    n_agents = len(agent_names)
    n_configs = len(config_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_configs)
    width = 0.8 / n_agents
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_agents))
    
    for i, agent_name in enumerate(agent_names):
        values = []
        for config_name in config_names:
            data = configs[config_name]
            if "bullwhip_per_agent_mean" in data and i < len(data["bullwhip_per_agent_mean"]):
                values.append(data["bullwhip_per_agent_mean"][i])
            else:
                values.append(0)
        
        offset = (i - n_agents/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=agent_name, color=colors[i])
    
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Bullwhip = 1.0')
    
    ax.set_xlabel('Weight Configuration', fontsize=12)
    ax.set_ylabel('Bullwhip Ratio', fontsize=12)
    ax.set_title(f'Bullwhip Effect by Agent/Echelon - {topology.upper()}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"bullwhip_comparison_{topology}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_latex_table(configs: Dict[str, Dict], topology: str, output_dir: str):
    """Generate LaTeX table for thesis."""
    sorted_configs = sorted(configs.items(), key=lambda x: x[1]["cost_weight"], reverse=True)
    
    latex = []
    latex.append(f"% Pareto Frontier Results - {topology}")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Multi-Objective Optimization Results - {topology.capitalize()} Topology}}")
    latex.append(f"\\label{{tab:pareto_{topology}}}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append("Configuration & $\\alpha$ & $\\beta$ & Total Cost & Fill Rate (\\%) & Bullwhip \\\\")
    latex.append("\\midrule")
    
    for config_name, data in sorted_configs:
        latex.append(
            f"{config_name.replace('_', ' ').title()} & "
            f"{data['cost_weight']:.1f} & "
            f"{data['service_weight']:.1f} & "
            f"{data['cost_mean']:.1f} $\\pm$ {data['cost_std']:.1f} & "
            f"{data['fill_rate_mean']*100:.2f} $\\pm$ {data['fill_rate_std']*100:.2f} & "
            f"{data['bullwhip_mean']:.3f} $\\pm$ {data['bullwhip_std']:.3f} \\\\"
        )
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    output_path = os.path.join(output_dir, f"table_pareto_{topology}.tex")
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"Saved: {output_path}")


def statistical_comparison(configs: Dict[str, Dict]):
    """Perform statistical tests between configurations."""
    if not HAS_SCIPY:
        print("Skipping statistical tests (scipy not installed)")
        return
    
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("(t-test comparing cost_focused vs service_focused)")
    print(f"{'='*80}")
    
    if "cost_focused" not in configs or "service_focused" not in configs:
        print("Required configs not found for comparison")
        return
    
    cost_results = configs["cost_focused"]["results"]
    service_results = configs["service_focused"]["results"]
    
    # Cost comparison
    cost_costs = [r.total_cost for r in cost_results]
    service_costs = [r.total_cost for r in service_results]
    t_stat, p_value = stats.ttest_ind(cost_costs, service_costs)
    print(f"\nTotal Cost:")
    print(f"  Cost-focused:    {np.mean(cost_costs):.1f} ± {np.std(cost_costs):.1f}")
    print(f"  Service-focused: {np.mean(service_costs):.1f} ± {np.std(service_costs):.1f}")
    print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    print(f"  {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")
    
    # Fill rate comparison
    cost_fr = [r.fill_rate for r in cost_results]
    service_fr = [r.fill_rate for r in service_results]
    t_stat, p_value = stats.ttest_ind(cost_fr, service_fr)
    print(f"\nFill Rate:")
    print(f"  Cost-focused:    {np.mean(cost_fr)*100:.2f}% ± {np.std(cost_fr)*100:.2f}%")
    print(f"  Service-focused: {np.mean(service_fr)*100:.2f}% ± {np.std(service_fr)*100:.2f}%")
    print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    print(f"  {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")


def export_to_csv(configs: Dict[str, Dict], topology: str, output_dir: str):
    """Export results to CSV for further analysis."""
    rows = []
    
    sorted_configs = sorted(configs.items(), key=lambda x: x[1]["cost_weight"], reverse=True)
    
    for config_name, data in sorted_configs:
        rows.append({
            "config": config_name,
            "cost_weight": data["cost_weight"],
            "service_weight": data["service_weight"],
            "cost_mean": data["cost_mean"],
            "cost_std": data["cost_std"],
            "fill_rate_mean": data["fill_rate_mean"],
            "fill_rate_std": data["fill_rate_std"],
            "bullwhip_mean": data["bullwhip_mean"],
            "bullwhip_std": data["bullwhip_std"],
            "n_seeds": data["n_seeds"],
        })
    
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        output_path = os.path.join(output_dir, f"pareto_results_{topology}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    else:
        # Basic CSV without pandas
        output_path = os.path.join(output_dir, f"pareto_results_{topology}.csv")
        with open(output_path, 'w') as f:
            headers = rows[0].keys()
            f.write(','.join(headers) + '\n')
            for row in rows:
                f.write(','.join(str(row[h]) for h in headers) + '\n')
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Pareto training results")
    parser.add_argument("--serial", type=str, default="results/pareto_serial",
                        help="Path to serial topology results")
    parser.add_argument("--network", type=str, default="results/pareto_network",
                        help="Path to network topology results")
    parser.add_argument("--output", type=str, default="results/analysis",
                        help="Output directory for figures and tables")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Agent names
    serial_agents = ["Retailer", "Distributor", "Factory"]
    network_agents = ["Retailer_1", "Retailer_2", "Distributor_1", "Distributor_2", "Factory_1", "Factory_2"]
    
    print("="*80)
    print("POST-TRAINING ANALYSIS FOR THESIS")
    print("="*80)
    
    # Analyze Serial
    if os.path.exists(args.serial):
        print(f"\n>>> Loading Serial results from {args.serial}")
        serial_results = load_results(args.serial)
        
        if serial_results:
            serial_configs = aggregate_by_config(serial_results)
            
            print_summary_table(serial_configs, "serial")
            print_per_agent_table(serial_configs, "serial", serial_agents)
            analyze_bullwhip_amplification(serial_configs, serial_agents)
            statistical_comparison(serial_configs)
            
            plot_pareto_frontier(serial_configs, "serial", args.output)
            plot_bullwhip_comparison(serial_configs, "serial", serial_agents, args.output)
            generate_latex_table(serial_configs, "serial", args.output)
            export_to_csv(serial_configs, "serial", args.output)
    else:
        print(f"Serial results not found at {args.serial}")
    
    # Analyze Network
    if os.path.exists(args.network):
        print(f"\n>>> Loading Network results from {args.network}")
        network_results = load_results(args.network)
        
        if network_results:
            network_configs = aggregate_by_config(network_results)
            
            print_summary_table(network_configs, "network")
            print_per_agent_table(network_configs, "network", network_agents)
            analyze_bullwhip_amplification(network_configs, network_agents)
            statistical_comparison(network_configs)
            
            plot_pareto_frontier(network_configs, "network", args.output)
            plot_bullwhip_comparison(network_configs, "network", network_agents, args.output)
            generate_latex_table(network_configs, "network", args.output)
            export_to_csv(network_configs, "network", args.output)
    else:
        print(f"Network results not found at {args.network}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Output saved to: {args.output}")
    print("="*80)
    
    # Summary of generated files
    print("\nGenerated files:")
    for f in os.listdir(args.output):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
