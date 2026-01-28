#!/bin/bash
# DoE Experiment Launcher for Debate System
# Comprehensive script to run Design of Experiments with W&B tracking

set -e  # Exit on error

# Configuration
WANDB_PROJECT="debates-doe"
OUTPUT_DIR="data"
ANALYSIS_DIR="analysis"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo ""
    echo "========================================================================"
    echo "  DoE Experiment Runner - Debate System with Misaligned LLMs"
    echo "========================================================================"
    echo ""
}

# Print help
print_help() {
    cat << EOF
Usage: ./run_doe.sh [EXPERIMENT_TYPE] [OPTIONS]

EXPERIMENT TYPES:
  pilot           Run pilot study (4 conditions) - quick validation
  fractional      Run fractional factorial (6 conditions) - default
  full            Run full factorial (36 conditions) - comprehensive
  sweep           Run W&B sweep (automated grid search)
  analyze         Analyze existing results
  
OPTIONS:
  --model MODEL   Specify model (default: glm-4.7-flash)
  --rounds N      Number of debate rounds (default: 4)
  --judge-runs N  Judge evaluations per condition (default: 10)
  --no-wandb      Disable W&B tracking
  --help          Show this help message

EXAMPLES:
  # Run pilot study
  ./run_doe.sh pilot

  # Run fractional factorial with custom settings
  ./run_doe.sh fractional --model llama2 --rounds 6

  # Run full factorial (warning: 36 conditions, takes time!)
  ./run_doe.sh full --judge-runs 20

  # Run W&B sweep (automated)
  ./run_doe.sh sweep

  # Analyze results from W&B
  ./run_doe.sh analyze

ENVIRONMENT VARIABLES:
  WANDB_API_KEY   Your W&B API token (required for W&B tracking)
  WANDB_GROUP     Custom group name for experiments
  WANDB_TAGS      Comma-separated tags for W&B

For more information, see README.md or CLI_USAGE.md
EOF
}

# Check dependencies
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    # Check uv
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}Error: uv not found. Install from https://docs.astral.sh/uv/${NC}"
        exit 1
    fi
    
    # Check if ollama is running (optional warning)
    if ! command -v ollama &> /dev/null; then
        echo -e "${YELLOW}Warning: ollama not found. Ensure Ollama is installed and running.${NC}"
    fi
    
    echo -e "${GREEN}✓ Dependencies OK${NC}"
}

# Run pilot study (4 conditions)
run_pilot() {
    echo -e "${GREEN}Running PILOT study (4 conditions)...${NC}"
    uv run python src/run_doe_batch.py \
        --mode manual \
        --design pilot \
        "$@"
}

# Run fractional factorial (6 conditions)
run_fractional() {
    echo -e "${GREEN}Running FRACTIONAL factorial (6 conditions)...${NC}"
    uv run python src/run_doe_batch.py \
        --mode manual \
        --design fractional \
        "$@"
}

# Run full factorial (36 conditions)
run_full() {
    echo -e "${YELLOW}Running FULL factorial (36 conditions)...${NC}"
    echo -e "${YELLOW}This will take significant time. Press Ctrl+C to cancel.${NC}"
    sleep 3
    uv run python src/run_doe_batch.py \
        --mode manual \
        --design full \
        "$@"
}

# Run W&B sweep
run_sweep() {
    echo -e "${GREEN}Running W&B SWEEP...${NC}"
    
    # Check for sweep config
    if [ ! -f "sweep_fractional.yaml" ]; then
        echo -e "${RED}Error: sweep_fractional.yaml not found${NC}"
        exit 1
    fi
    
    uv run python src/run_doe_batch.py \
        --mode sweep \
        --sweep-config sweep_fractional.yaml \
        "$@"
}

# Analyze results
run_analysis() {
    echo -e "${GREEN}Analyzing results...${NC}"
    
    # Get latest W&B group if not specified
    if [ -z "$WANDB_GROUP" ]; then
        echo -e "${YELLOW}No WANDB_GROUP specified. Analyzing all runs in project.${NC}"
        uv run python src/analyze_doe_results.py \
            --source wandb \
            --project "$WANDB_PROJECT" \
            --export "${ANALYSIS_DIR}/results.csv" \
            --report "${ANALYSIS_DIR}/report.txt" \
            --plot \
            --output-dir "$ANALYSIS_DIR" \
            "$@"
    else
        echo -e "${GREEN}Analyzing W&B group: $WANDB_GROUP${NC}"
        uv run python src/analyze_doe_results.py \
            --source wandb \
            --project "$WANDB_PROJECT" \
            --group "$WANDB_GROUP" \
            --export "${ANALYSIS_DIR}/results.csv" \
            --report "${ANALYSIS_DIR}/report.txt" \
            --plot \
            --output-dir "$ANALYSIS_DIR" \
            "$@"
    fi
    
    echo -e "${GREEN}✓ Analysis complete! Check ${ANALYSIS_DIR}/ for results${NC}"
}

# Main script
main() {
    print_banner
    
    # Parse command
    EXPERIMENT_TYPE="${1:-fractional}"
    shift || true
    
    # Show help
    if [ "$EXPERIMENT_TYPE" = "--help" ] || [ "$EXPERIMENT_TYPE" = "-h" ]; then
        print_help
        exit 0
    fi
    
    # Check dependencies
    check_dependencies
    
    # Set default W&B group if not set
    if [ -z "$WANDB_GROUP" ]; then
        export WANDB_GROUP="DoE-${EXPERIMENT_TYPE}-$(date +%Y%m%d-%H%M%S)"
        echo -e "${YELLOW}W&B group: $WANDB_GROUP${NC}"
    fi
    
    # Run experiment
    case "$EXPERIMENT_TYPE" in
        pilot)
            run_pilot "$@"
            ;;
        fractional)
            run_fractional "$@"
            ;;
        full)
            run_full "$@"
            ;;
        sweep)
            run_sweep "$@"
            ;;
        analyze)
            run_analysis "$@"
            ;;
        *)
            echo -e "${RED}Error: Unknown experiment type '$EXPERIMENT_TYPE'${NC}"
            echo ""
            print_help
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}========================================================================"
    echo -e "  Experiment Complete!"
    echo -e "========================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. View results in W&B: https://wandb.ai/${WANDB_PROJECT}"
    echo "  2. Run analysis: ./run_doe.sh analyze"
    echo "  3. Check local data: ${OUTPUT_DIR}/"
    echo ""
}

# Run main
main "$@"
