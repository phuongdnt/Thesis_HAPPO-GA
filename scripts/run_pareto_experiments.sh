#!/bin/bash
# =============================================================================
# PARETO EXPERIMENTS AUTOMATION SCRIPT
# =============================================================================
# This script automates the entire Pareto frontier generation process:
# 1. Train models with different weight configurations
# 2. Evaluate trained models
# 3. Generate Pareto frontier analysis and plots
#
# Usage:
#   chmod +x scripts/run_pareto_experiments.sh
#   ./scripts/run_pareto_experiments.sh
#
# Requirements:
#   - Python 3.8+
#   - PyTorch
#   - matplotlib (for visualization)
#   - scipy (for statistical tests)
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs/pareto_${TIMESTAMP}"
SERIAL_CONFIG="${PROJECT_DIR}/configs/pareto_serial.yaml"
NETWORK_CONFIG="${PROJECT_DIR}/configs/pareto_network.yaml"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_DIR}/results/pareto_serial"
mkdir -p "${PROJECT_DIR}/results/pareto_network"

echo "=============================================="
echo "PARETO FRONTIER EXPERIMENTS"
echo "=============================================="
echo "Project directory: ${PROJECT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Function to run experiment with timeout and logging
run_experiment() {
    local config=$1
    local topology=$2
    local log_file="${LOG_DIR}/${topology}_training.log"
    
    echo "Starting ${topology} topology training..."
    echo "Config: ${config}"
    echo "Log: ${log_file}"
    
    cd "${PROJECT_DIR}"
    
    # Run training with output logged
    if [ "${topology}" = "network" ]; then
        python train_pareto_network.py \
            --config "${config}" \
            2>&1 | tee "${log_file}"
    else
        python train_pareto.py \
            --config "${config}" \
            2>&1 | tee "${log_file}"
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ ${topology} training completed successfully"
    else
        echo "✗ ${topology} training failed"
        return 1
    fi
}

# Function to run analysis
run_analysis() {
    local log_file="${LOG_DIR}/analysis.log"
    
    echo ""
    echo "Running Pareto analysis..."
    
    python analyze_pareto.py \
        --serial "${PROJECT_DIR}/results/pareto_serial" \
        --network "${PROJECT_DIR}/results/pareto_network" \
        --output "${PROJECT_DIR}/results/analysis" \
        2>&1 | tee "${log_file}"
    echo "✓ Analysis completed"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo ""
echo "=============================================="
echo "PHASE 1: SERIAL TOPOLOGY EXPERIMENTS"
echo "=============================================="

if [ -f "${SERIAL_CONFIG}" ]; then
    run_experiment "${SERIAL_CONFIG}" "serial"
else
    echo "Warning: Serial config not found at ${SERIAL_CONFIG}"
fi

echo ""
echo "=============================================="
echo "PHASE 2: NETWORK TOPOLOGY EXPERIMENTS"
echo "=============================================="

if [ -f "${NETWORK_CONFIG}" ]; then
    run_experiment "${NETWORK_CONFIG}" "network"
else
    echo "Warning: Network config not found at ${NETWORK_CONFIG}"
    echo "Skipping network topology experiments"
fi

echo ""
echo "=============================================="
echo "PHASE 3: PARETO ANALYSIS"
echo "=============================================="

run_analysis

echo ""
echo "=============================================="
echo "EXPERIMENT SUMMARY"
echo "=============================================="
echo ""
echo "Results locations:"
echo "  Serial:   ${PROJECT_DIR}/results/pareto_serial/"
echo "  Network:  ${PROJECT_DIR}/results/pareto_network/"
echo "  Analysis: ${PROJECT_DIR}/results/analysis/"
echo ""
echo "Generated files:"
echo "  - pareto_results.json: Raw experiment data"
echo "  - pareto_frontier.png: Pareto frontier plot"
echo "  - metric_comparison.png: Multi-metric bar charts"
echo "  - pareto_table.tex: LaTeX table for thesis"
echo "  - analysis_report.txt: Summary report"
echo ""
echo "Logs: ${LOG_DIR}"
echo ""
echo "=============================================="
echo "COMPLETE"
echo "=============================================="
