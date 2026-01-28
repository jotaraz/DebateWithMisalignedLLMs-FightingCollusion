# DoE Quick Reference Card

## Fastest Path to Results

```bash
# 1. Login to W&B (one-time setup)
uv run wandb login
# Enter token: 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

# 2. Run pilot study (4 conditions, fastest)
./run_doe.sh pilot

# 3. View results
# Browser: https://wandb.ai/debates-doe
# Or: ./run_doe.sh analyze
```

## All Experiment Types

| Command | Conditions | Time | Description |
|---------|------------|------|-------------|
| `./run_doe.sh pilot` | 4 | ~10 min | Quick validation |
| `./run_doe.sh fractional` | 6 | ~15 min | Standard DoE (Debater 0 only) |
| `./run_doe.sh full` | 36 | ~90 min | Comprehensive (both debaters) |
| `./run_doe.sh sweep` | varies | varies | W&B automated sweep |
| `./run_doe.sh analyze` | - | ~1 min | Generate analysis reports |

## Custom Parameters

```bash
# Different model
./run_doe.sh fractional --model llama2

# More rounds
./run_doe.sh fractional --rounds 6

# Higher precision
./run_doe.sh fractional --judge-runs 20

# Combine options
./run_doe.sh fractional --model llama2 --rounds 6 --judge-runs 20
```

## Experimental Factors

### Primary (DoE Variables)
- **public_strength**: 0=admit defeat, 1=normal, 2=replacement threat
- **hidden_strength**: 0=no hidden agenda, 1=hidden agenda

### Fixed (Default)
- **Topic**: cats vs dogs
- **Model**: glm-4.7-flash
- **Rounds**: 4
- **Judge runs**: 10

## File Outputs

```
data/
├── cd00dd00.txt          # Conversations
├── cd01dd00.txt
├── cd10dd00.txt
├── ...
└── judge_results.txt     # Aggregated verdicts

analysis/
├── results.csv           # Full data table
├── report.txt            # Summary statistics
├── win_rates_by_public_strength.png
└── win_rate_heatmap.png
```

## Naming Convention

`cd{pub0}{hid0}dd{pub1}{hid1}.txt`
- Example: `cd21dd00.txt`
  - Debater 0: public=2 (threatened), hidden=1 (misaligned)
  - Debater 1: public=0 (admit defeat), hidden=0 (aligned)

## W&B Dashboard

**URL:** https://wandb.ai/debates-doe

**Filter by:**
- Group: `DoE-fractional-20260127-153045`
- Tags: `DoE`, `fractional`, `full`, `model_glm4`

## Common Tasks

### Run Single Experiment
```bash
uv run python src/cli_debate.py \
    --public-agenda-0 cats \
    --public-agenda-1 dogs \
    --public-strength-0 2 \
    --hidden-strength-0 1 \
    --wandb
```

### Analyze Specific Group
```bash
WANDB_GROUP="DoE-fractional-20260127" ./run_doe.sh analyze
```

### Export Results to CSV
```bash
uv run python src/analyze_doe_results.py \
    --source wandb \
    --project debates-doe \
    --export my_results.csv
```

### Run Without W&B
```bash
./run_doe.sh fractional --no-wandb
```

## Environment Variables

```bash
# W&B API key (required for tracking)
export WANDB_API_KEY="wandb_v1_..."

# Custom experiment group
export WANDB_GROUP="MyExperiment-2026"

# Add custom tags
export WANDB_TAGS="validation,test,important"

# Offline mode (sync later)
export WANDB_MODE="offline"
```

## Troubleshooting

**Issue:** W&B authentication failed
```bash
uv run wandb login
```

**Issue:** Ollama not responding
```bash
ollama list
ollama run glm-4.7-flash "test"
```

**Issue:** Script permission denied
```bash
chmod +x run_doe.sh
```

**Issue:** Long experiment - run in background
```bash
nohup ./run_doe.sh full > experiment.log 2>&1 &
tail -f experiment.log
```

## Python API

### Run Batch Programmatically

```python
from src.run_doe_batch import run_manual_batch

run_manual_batch(
    design="fractional",
    model_0="glm-4.7-flash",
    rounds=4,
    judge_runs=10,
    use_wandb=True,
    wandb_group="MyExperiment"
)
```

### Analyze Programmatically

```python
from src.analyze_doe_results import analyze_wandb_results

df = analyze_wandb_results(
    project="debates-doe",
    group="DoE-fractional-20260127"
)

print(df.describe())
```

## Statistical Analysis

```python
import pandas as pd
from scipy import stats

# Load results
df = pd.read_csv('analysis/results.csv')

# ANOVA for main effect
groups = df.groupby('public_incentive_strength_0')['win_rate_0'].apply(list)
f_stat, p_value = stats.f_oneway(*groups)
print(f"F={f_stat:.3f}, p={p_value:.4f}")

# Effect size (Cohen's d)
group_0 = df[df['public_incentive_strength_0'] == 0]['win_rate_0']
group_2 = df[df['public_incentive_strength_0'] == 2]['win_rate_0']
pooled_std = df['win_rate_0'].std()
cohens_d = (group_2.mean() - group_0.mean()) / pooled_std
print(f"Cohen's d = {cohens_d:.3f}")
```

## Tips

1. **Start small:** Always run pilot before full factorial
2. **Monitor W&B:** Check dashboard during experiments
3. **Save API key:** Export WANDB_API_KEY in ~/.bashrc
4. **Use groups:** Organize experiments with descriptive group names
5. **Replicate:** Run important conditions multiple times for reliability
6. **Document:** Add notes to W&B runs for future reference

## Getting Help

- **DoE Guide:** [DOE_GUIDE.md](DOE_GUIDE.md)
- **CLI Usage:** [CLI_USAGE.md](CLI_USAGE.md)
- **README:** [README.md](README.md)
- **W&B Docs:** https://docs.wandb.ai/
