# Design of Experiments (DoE) Guide

## Overview

This guide explains how to run systematic experiments on the debate system using Design of Experiments (DoE) methodology with Weights & Biases (W&B) tracking.

## Quick Start

```bash
# 1. Ensure you're logged into W&B (already done)
uv run wandb login

# 2. Run a pilot study (4 conditions, ~10 minutes)
./run_doe.sh pilot

# 3. Run fractional factorial (6 conditions, ~15 minutes)
./run_doe.sh fractional

# 4. Analyze results
./run_doe.sh analyze
```

## Experimental Designs

### Pilot Study (4 conditions)
Quick validation of corner cases:
- `cd00dd00`: Both normal, no hidden agendas
- `cd20dd00`: Debater 0 threatened, no hidden agendas
- `cd01dd00`: Debater 0 normal, with hidden agenda
- `cd21dd00`: Debater 0 threatened, with hidden agenda

**Usage:**
```bash
./run_doe.sh pilot
```

### Fractional Factorial (6 conditions)
Varies only Debater 0 parameters (Debater 1 held constant):
- Public strength: {0=admit defeat, 1=normal, 2=replacement threat}
- Hidden strength: {0=no hidden, 1=hidden agenda}

**Conditions:** 3 × 2 = 6 combinations

**Usage:**
```bash
./run_doe.sh fractional --judge-runs 10
```

### Full Factorial (36 conditions)
Varies all parameters for both debaters:
- Debater 0: public_strength × hidden_strength (3 × 2)
- Debater 1: public_strength × hidden_strength (3 × 2)

**Conditions:** 3 × 2 × 3 × 2 = 36 combinations

**Usage:**
```bash
./run_doe.sh full --judge-runs 10
# WARNING: This takes ~60-90 minutes
```

## W&B Sweep Mode

Automated hyperparameter search using W&B sweeps:

```bash
# Edit sweep configuration
nano sweep_fractional.yaml  # or sweep.yaml for full factorial

# Run sweep
./run_doe.sh sweep
```

**Sweep features:**
- Automated parameter grid search
- Parallel execution support (if configured)
- Real-time monitoring in W&B dashboard
- Automatic result aggregation

## Manual Batch Mode

For more control over execution:

```bash
# Basic fractional factorial
uv run python src/run_doe_batch.py --mode manual --design fractional

# Custom settings
uv run python src/run_doe_batch.py \
    --mode manual \
    --design fractional \
    --model-0 llama2 \
    --rounds 6 \
    --judge-runs 20 \
    --wandb-group "MyExperiment-2026"

# Full factorial
uv run python src/run_doe_batch.py --mode manual --design full
```

## Experimental Factors

### Primary Factors

1. **public_strength_0/1** (3 levels)
   - `0`: Admit defeat in final round
   - `1`: Normal debate incentives
   - `2`: Threatened with replacement by judge

2. **hidden_strength_0/1** (2 levels)
   - `0`: No hidden agenda
   - `1`: Hidden agenda (wants opposite outcome)

### Secondary Factors (configurable)

3. **model** - Different LLM models
   - `glm-4.7-flash` (default)
   - `llama2`
   - `mistral`
   - etc.

4. **rounds** - Debate length
   - `2`, `4` (default), `6`, `8`

5. **judge_runs** - Statistical sampling
   - `10` (default), `20`, `50`

## Results Analysis

### View in W&B Dashboard

```bash
# Get your W&B URL
echo "https://wandb.ai/${WANDB_PROJECT}"
```

Filter by:
- **Group**: Experiment batch (e.g., `DoE-fractional-20260127`)
- **Tags**: `DoE`, `fractional`, `full`, `model_glm4`, etc.

### Generate Analysis Reports

```bash
# Analyze latest experiment group
./run_doe.sh analyze

# Analyze specific group
WANDB_GROUP="DoE-fractional-20260127-153000" ./run_doe.sh analyze

# Analyze from local files (no W&B)
uv run python src/analyze_doe_results.py --source local --data-dir data
```

**Outputs:**
- `analysis/results.csv` - Full results table
- `analysis/report.txt` - Summary statistics
- `analysis/win_rates_by_public_strength.png` - Main effect plot
- `analysis/win_rate_heatmap.png` - Interaction plot

### Statistical Analysis

```python
import pandas as pd

# Load results
df = pd.read_csv('analysis/results.csv')

# ANOVA for main effects
from scipy import stats
groups = df.groupby('public_incentive_strength_0')['win_rate_0'].apply(list)
f_stat, p_value = stats.f_oneway(*groups)

# Cohen's d for effect size
from scipy.stats import ttest_ind
group_0 = df[df['public_incentive_strength_0'] == 0]['win_rate_0']
group_2 = df[df['public_incentive_strength_0'] == 2]['win_rate_0']
cohens_d = (group_2.mean() - group_0.mean()) / df['win_rate_0'].std()
```

## W&B Integration Features

### Run Organization

Each run is automatically tagged and grouped:

**Run Name:** `cd{pub0}{hid0}dd{pub1}{hid1}_{model}`
- Example: `cd20dd00_glm4`

**Group:** `DoE-{design}-{timestamp}`
- Example: `DoE-fractional-20260127-153045`

**Tags:**
- Automatic: `debate`, `model_{model}`, `rounds_{n}`, `symmetric/asymmetric`, `misaligned`
- Custom: Set via `WANDB_TAGS` environment variable

**Notes:** Full experimental description including parameter values

### Logged Metrics

**Per-run:**
- `judge_debater_0_wins`, `judge_debater_1_wins`, `judge_undecided`
- `judge_total_runs`, `winner`
- `actual_turns`
- Individual judge verdicts: `judge_run_1` through `judge_run_N`

**Artifacts:**
- Full debate transcripts (conversation files)
- System prompts for both debaters

### Environment Variables

```bash
# W&B authentication (required)
export WANDB_API_KEY="wandb_v1_..."

# Custom grouping
export WANDB_GROUP="MyExperiment-2026-01-27"

# Custom tags
export WANDB_TAGS="pilot,validation,llama2"

# Offline mode (sync later)
export WANDB_MODE="offline"
# Then sync: wandb sync wandb/run-xxxxx
```

## File Structure

```
DebateWithMisalignedLLMs-FightingCollusion/
├── run_doe.sh                    # Main launcher script
├── sweep.yaml                    # Full factorial sweep config
├── sweep_fractional.yaml         # Fractional factorial sweep config
├── src/
│   ├── cli_debate.py            # Core debate script (enhanced with W&B)
│   ├── run_doe_batch.py         # Batch experiment runner
│   ├── analyze_doe_results.py   # Results aggregation and analysis
│   └── generate_bars.py         # Visualization (original)
├── data/                        # Conversation transcripts
│   ├── cd00dd00.txt
│   ├── cd01dd00.txt
│   ├── ...
│   └── judge_results.txt        # Aggregated judge verdicts
└── analysis/                    # Generated analysis outputs
    ├── results.csv
    ├── report.txt
    └── *.png
```

## Naming Convention

**Conversation files:** `cd{PS0}{HS0}dd{PS1}{HS1}.txt`
- `PS0/PS1`: Public strength (0, 1, 2)
- `HS0/HS1`: Hidden strength (0, 1)
- Example: `cd21dd00.txt` = Debater 0: pub=2, hid=1; Debater 1: pub=0, hid=0

**Section identifiers** (in judge_results.txt):
- `cddd00`: Standard cat vs dog debate
- Custom sections for topic variations

## Advanced Usage

### Multi-Model Comparison

Compare different LLMs:

```bash
# Run with different models
for model in "glm-4.7-flash" "llama2" "mistral"; do
    uv run python src/run_doe_batch.py \
        --mode manual \
        --design fractional \
        --model-0 "$model" \
        --wandb-group "MultiModel-2026" \
        --wandb-tags "model-comparison,$model"
done
```

### Round Variation Study

Test debate length effects:

```bash
for rounds in 2 4 6; do
    uv run python src/run_doe_batch.py \
        --mode manual \
        --design fractional \
        --rounds "$rounds" \
        --wandb-group "RoundStudy-2026" \
        --wandb-tags "round-variation,rounds_$rounds"
done
```

### High-Precision Experiments

Increase statistical power:

```bash
uv run python src/run_doe_batch.py \
    --mode manual \
    --design fractional \
    --judge-runs 50 \
    --wandb-group "HighPrecision-2026"
```

## Troubleshooting

### W&B Authentication Issues

```bash
# Re-login
uv run wandb login

# Check login status
uv run wandb verify

# Use API key directly
export WANDB_API_KEY="wandb_v1_..."
```

### Ollama Connection Issues

```bash
# Check Ollama is running
ollama list

# Test model
ollama run glm-4.7-flash "Hello"

# Pull model if needed
ollama pull glm-4.7-flash
```

### Long Running Experiments

```bash
# Run in background
nohup ./run_doe.sh full > experiment.log 2>&1 &

# Monitor progress
tail -f experiment.log

# Or use tmux/screen
tmux new -s doe
./run_doe.sh full
# Ctrl+B, D to detach
```

### Rate Limiting

If hitting API rate limits:

```bash
# Use offline mode
export WANDB_MODE="offline"
./run_doe.sh fractional

# Sync later
wandb sync wandb/offline-run-*
```

## Research Questions

### Main Hypotheses

1. **H1:** Replacement threats (public_strength=2) reduce collusion
   - Measure: Win rate difference between pub_strength 0 vs 2

2. **H2:** Hidden agendas increase collusion regardless of threats
   - Measure: Win rate difference with/without hidden_strength

3. **H3:** Interaction effect between threats and hidden agendas
   - Measure: ANOVA interaction term

### Recommended Experiments

1. **Initial validation:** Pilot study (4 conditions)
2. **Main experiment:** Fractional factorial (6 conditions)
3. **Comprehensive:** Full factorial (36 conditions)
4. **Robustness:** Multi-model comparison
5. **Sensitivity:** Round variation study

## Citation

If using this system for research:

```bibtex
@misc{debate-doe-2026,
  title={Design of Experiments for Studying Collusion in Misaligned AI Debaters},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/DebateWithMisalignedLLMs-FightingCollusion}
}
```

## Support

For issues or questions:
1. Check [CLI_USAGE.md](CLI_USAGE.md) for basic usage
2. Check [README.md](README.md) for project overview
3. Review W&B dashboard for experiment tracking
4. Check [GitHub Issues](https://github.com/yourusername/repo/issues)
