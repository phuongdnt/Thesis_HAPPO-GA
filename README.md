# HAPPO + GA Hybrid for Multi-Echelon Inventory Optimization

This repository contains the code for the undergraduate thesis:
**"Multi-Agent Deep Reinforcement Learning for Multi-Echelon Inventory
Systems Under Demand Uncertainty"**.

The approach combines Heterogeneous-Agent Proximal Policy Optimization
(HAPPO) with a Genetic Algorithm (GA) lot-sizing heuristic.  Training
uses Pareto multi-objective optimization with five weight configurations
(α, β) to explore cost–service trade-offs.  Two supply chain topologies
are supported: a 3-echelon serial chain and a 6-agent network topology.

---

## Project Structure

```
├── configs/                    # Experiment configurations
│   ├── pareto_serial.yaml      #   Serial topology (3 agents)
│   └── pareto_network.yaml     #   Network topology (6 agents)
│
├── envs/                       # Supply chain simulation environments
│   ├── base_env.py             #   Abstract base class
│   ├── serial_env.py           #   Serial multi-echelon chain
│   ├── network_env.py          #   General network topology
│   ├── reward_functions.py     #   Cost & service metric functions
│   └── vec_env.py              #   Parallel vectorized env wrapper
│
├── agents/                     # HAPPO agent implementation
│   ├── happo_agent.py          #   Main HAPPO trainer
│   ├── policy_networks.py      #   MLP actor & critic networks
│   ├── centralized_critic.py   #   Centralized critic (CTDE)
│   └── replay_buffer.py        #   Rollout buffer
│
├── lot_sizing/                 # Heuristic lot-sizing modules
│   ├── ga_lotsizing.py         #   Adaptive genetic algorithm
│   └── hybrid_planner.py       #   RL + GA hybrid planner
│
├── utils/                      # Helpers
│   ├── logger.py               #   Logging setup
│   └── metrics.py              #   Bullwhip, service level metrics
│
├── test_data/                  # Evaluation demand sequences
│   ├── config.yaml             #   Environment parameter defaults
│   ├── test_demand_merton/     #   Merton jump-diffusion demand (30 files)
│   └── test_demand_stationary/ #   Stationary Poisson demand (30 files)
│
├── results/                    # Training outputs
│   ├── pareto_serial/          #   Serial topology results
│   └── pareto_network/         #   Network topology results
│
├── train_pareto.py             # Train: serial topology
├── train_pareto_network.py     # Train: network topology
├── evaluate_all.py             # Evaluate all configs + baselines
├── analyze_pareto.py           # Post-training Pareto analysis & plots
├── baselines.py                # Classical baselines (Base-stock, (s,S))
├── data_loader.py              # Demand data loading & generation
├── reward_utils.py             # Multi-objective reward computation
├── training_utils.py           # Early stopping, LR scheduling, monitoring
├── scripts/
│   └── run_pareto_experiments.sh  # Automation script
└── requirements.txt
```

## Getting Started

### 1. Install dependencies

Requires **Python 3.8+** and PyTorch.

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train (serial topology)

```bash
python train_pareto.py --config configs/pareto_serial.yaml
```

This trains five Pareto weight configurations (cost-focused → service-focused)
across three random seeds.  Results are saved to `results/pareto_serial/`.

### 3. Train (network topology)

```bash
python train_pareto_network.py --config configs/pareto_network.yaml
```

### 4. Evaluate & compare with baselines

```bash
python evaluate_all.py --serial results/pareto_serial --network results/pareto_network
```

Compares HAPPO+GA with Base-stock and (s,S) policies.  Outputs LaTeX tables
and CSV summaries.

### 5. Analyze Pareto frontier

```bash
python analyze_pareto.py --serial results/pareto_serial --network results/pareto_network
```

Generates Pareto frontier plots, per-agent metrics, bullwhip analysis,
and statistical significance tests.

### 6. Run everything automatically

```bash
chmod +x scripts/run_pareto_experiments.sh
./scripts/run_pareto_experiments.sh
```

## Key Design Choices

- **Reward**: `R_i = α × raw_cost_i + β × fill_rate_i × γ` where (α, β)
  weights are varied across five Pareto configurations.
- **GA parameters**: population = 15, generations = 20, mutation rate = 0.1
  (configurable in YAML).
- **Early stopping**: patience-based with LR reduction on plateau.
- **Vectorized training**: true parallel execution via `multiprocessing`.

## Pareto Weight Configurations

| Config          | α (cost) | β (service) |
|-----------------|----------|-------------|
| cost_focused    | 1.0      | 0.0         |
| cost_leaning    | 0.7      | 0.3         |
| balanced        | 0.5      | 0.5         |
| service_leaning | 0.3      | 0.7         |
| service_focused | 0.0      | 1.0         |

## Dependencies

- PyTorch, NumPy, SciPy, Matplotlib, PyYAML, tqdm, cloudpickle
