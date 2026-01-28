# DoE Implementation Summary

## ✅ Implementation Complete

A comprehensive Design of Experiments (DoE) framework has been implemented for the debate system with full Weights & Biases integration.

## 📁 Files Created

### Configuration Files
- ✅ **sweep.yaml** - Full factorial W&B sweep configuration (36 conditions)
- ✅ **sweep_fractional.yaml** - Fractional factorial sweep configuration (6 conditions)

### Core Scripts
- ✅ **src/run_doe_batch.py** - Batch experiment runner with manual and sweep modes
- ✅ **src/analyze_doe_results.py** - Results aggregation and statistical analysis
- ✅ **run_doe.sh** - Convenient launcher script with multiple experiment types

### Documentation
- ✅ **DOE_GUIDE.md** - Comprehensive DoE methodology guide
- ✅ **DOE_QUICKREF.md** - Quick reference card for common tasks
- ✅ **README.md** - Updated with DoE quick start section

### Enhanced Existing Files
- ✅ **src/cli_debate.py** - Enhanced W&B integration with:
  - Descriptive run names (e.g., `cd20dd00_glm4`)
  - Automatic grouping and tagging
  - Experiment notes
  - Artifact logging for conversation files
  - Per-judge-run metrics tracking

## 🎯 Experimental Designs Available

| Design | Conditions | Factors Varied | Time Estimate |
|--------|------------|----------------|---------------|
| **Pilot** | 4 | Corner cases only | ~10 min |
| **Fractional** | 6 | Debater 0 only (3×2) | ~15 min |
| **Full** | 36 | Both debaters (3×2×3×2) | ~90 min |
| **Sweep** | Configurable | W&B automated | Varies |

## 🚀 Quick Start Commands

```bash
# Run pilot study (recommended first step)
./run_doe.sh pilot

# Run standard fractional factorial
./run_doe.sh fractional

# Run full factorial (comprehensive)
./run_doe.sh full

# Run W&B sweep (automated)
./run_doe.sh sweep

# Analyze results
./run_doe.sh analyze
```

## 🎨 W&B Integration Features

### Run Organization
- **Names:** `cd{pub0}{hid0}dd{pub1}{hid1}_{model}`
- **Groups:** `DoE-{design}-{timestamp}`
- **Tags:** Automatic + custom via environment variables
- **Notes:** Full experimental description

### Logged Metrics
- Judge verdicts (wins, undecided)
- Individual judge runs (per-run verdict tracking)
- Debate metadata (turns, models, parameters)

### Artifacts
- Full debate transcripts
- System prompts
- Configuration snapshots

### Environment Variables
```bash
WANDB_API_KEY     # Authentication (required)
WANDB_GROUP       # Custom experiment grouping
WANDB_TAGS        # Custom tags (comma-separated)
WANDB_MODE        # online/offline mode
```

## 📊 Analysis Capabilities

### From W&B API
```bash
uv run python src/analyze_doe_results.py \
    --source wandb \
    --project debates-doe \
    --group "DoE-fractional-20260127" \
    --export results.csv \
    --plot \
    --output-dir analysis
```

### From Local Files
```bash
uv run python src/analyze_doe_results.py \
    --source local \
    --data-dir data \
    --export results.csv
```

### Outputs Generated
- **results.csv** - Full data table
- **report.txt** - Summary statistics by condition
- **win_rates_by_public_strength.png** - Main effect plot
- **win_rate_heatmap.png** - Interaction heatmap

## 🔧 Experimental Factors

### Primary Factors (DoE Variables)
1. **public_strength_0** (3 levels: 0, 1, 2)
   - 0: Admit defeat in final round
   - 1: Normal debate incentives
   - 2: Replacement threat

2. **hidden_strength_0** (2 levels: 0, 1)
   - 0: No hidden agenda
   - 1: Hidden agenda (misaligned)

3. **public_strength_1** (3 levels: 0, 1, 2) - Full factorial only
4. **hidden_strength_1** (2 levels: 0, 1) - Full factorial only

### Secondary Factors (Configurable)
- **model**: LLM model selection (glm-4.7-flash, llama2, mistral, etc.)
- **rounds**: Debate length (2, 4, 6, 8)
- **judge_runs**: Statistical sampling (10, 20, 50)
- **topic**: Debate subject (cats/dogs, custom agendas)

## 🎛️ Usage Modes

### 1. Manual Batch Mode
Sequential execution with full control:
```bash
uv run python src/run_doe_batch.py \
    --mode manual \
    --design fractional \
    --model-0 glm-4.7-flash \
    --rounds 4 \
    --judge-runs 10
```

### 2. W&B Sweep Mode
Automated parameter grid search:
```bash
uv run python src/run_doe_batch.py \
    --mode sweep \
    --sweep-config sweep_fractional.yaml
```

### 3. Launcher Script Mode
Convenient high-level interface:
```bash
./run_doe.sh fractional --model llama2 --rounds 6
```

## 📈 Statistical Analysis Example

```python
import pandas as pd
from scipy import stats

# Load results
df = pd.read_csv('analysis/results.csv')

# Test main effect of public strength
groups = df.groupby('public_incentive_strength_0')['win_rate_0'].apply(list)
f_stat, p_value = stats.f_oneway(*groups)
print(f"ANOVA: F={f_stat:.3f}, p={p_value:.4f}")

# Calculate effect size
group_0 = df[df['public_incentive_strength_0'] == 0]['win_rate_0']
group_2 = df[df['public_incentive_strength_0'] == 2]['win_rate_0']
cohens_d = (group_2.mean() - group_0.mean()) / df['win_rate_0'].std()
print(f"Effect size (Cohen's d): {cohens_d:.3f}")
```

## 🔬 Research Workflow

### Recommended Sequence
1. **Pilot Study** (4 conditions) - Validate setup and check for obvious issues
2. **Fractional Factorial** (6 conditions) - Main experiment, varies Debater 0 only
3. **Analysis** - Check results, compute statistics, generate plots
4. **Full Factorial** (36 conditions) - If needed for interaction effects
5. **Robustness Checks** - Multi-model comparison, round variation studies

### Example Complete Workflow
```bash
# Step 1: Pilot
./run_doe.sh pilot
./run_doe.sh analyze

# Step 2: Main experiment
WANDB_GROUP="MainStudy-2026" ./run_doe.sh fractional
./run_doe.sh analyze

# Step 3: Generate publication plots
uv run python src/analyze_doe_results.py \
    --source wandb \
    --group "MainStudy-2026" \
    --plot \
    --export publication_results.csv

# Step 4: Statistical analysis in Python/R
# (Use publication_results.csv)
```

## 📖 Documentation Structure

- **DOE_GUIDE.md** - Comprehensive guide (methodology, usage, troubleshooting)
- **DOE_QUICKREF.md** - Quick reference card (commands, examples)
- **CLI_USAGE.md** - Single experiment CLI guide (existing)
- **README.md** - Project overview with DoE quick start (updated)

## ✨ Key Improvements to cli_debate.py

1. **Enhanced wandb.init():**
   - Descriptive run names based on parameters
   - Automatic group assignment (date-based or custom)
   - Smart tagging (symmetric/asymmetric, misaligned, model, rounds)
   - Rich notes with full parameter description

2. **Artifact Logging:**
   - Conversation transcripts logged as W&B artifacts
   - Preserves full debate history with metadata

3. **Per-Judge Metrics:**
   - Individual judge verdicts logged (judge_run_1, judge_run_2, ...)
   - Enables within-condition variability analysis

4. **Environment Variable Support:**
   - WANDB_GROUP for custom grouping
   - WANDB_TAGS for custom tagging
   - Flexible organization of experiments

## 🎓 Educational Value

The implementation serves as a complete example of:
- Design of Experiments methodology in ML research
- Experiment tracking with W&B
- Systematic parameter exploration
- Result aggregation and statistical analysis
- Reproducible research practices

## 🔄 Next Steps (Optional Enhancements)

### Immediate Use
1. Run pilot study to validate setup
2. Execute fractional factorial for main results
3. Analyze and visualize findings

### Future Enhancements (if needed)
1. Add parallelization for faster batch execution
2. Implement Bayesian optimization for parameter search
3. Add more visualization types (violin plots, box plots)
4. Integrate with R for advanced statistical analysis
5. Add power analysis for sample size determination
6. Create automated report generation (LaTeX/PDF)

## 📊 Expected Outputs

### Per Run
- Conversation transcript (data/cd{XX}dd{YY}.txt)
- Judge results entry (data/judge_results.txt)
- W&B run with metrics, config, artifacts

### Per Experiment Batch
- W&B group with all runs
- Organized by tags and metadata
- Traceable and reproducible

### Analysis Phase
- CSV export of all results
- Summary statistics report
- Visualization plots (PNG)
- Statistical test results

## 🎯 Success Criteria

✅ All implementation tasks completed
✅ Scripts tested and working
✅ Documentation comprehensive and clear
✅ W&B integration functional
✅ Analysis pipeline operational
✅ Quick start examples provided

## 💡 Tips for Users

1. **Start with pilot:** Always validate with 4 conditions first
2. **Monitor W&B:** Check dashboard during experiments for issues
3. **Use groups:** Organize experiments with descriptive group names
4. **Save tokens:** Export WANDB_API_KEY in shell profile
5. **Document:** Add notes to W&B runs for future reference
6. **Replicate:** Run important conditions multiple times
7. **Analyze early:** Don't wait for all experiments to complete

## 🆘 Support Resources

- **Help command:** `./run_doe.sh --help`
- **DoE Guide:** [DOE_GUIDE.md](DOE_GUIDE.md)
- **Quick Ref:** [DOE_QUICKREF.md](DOE_QUICKREF.md)
- **CLI Guide:** [CLI_USAGE.md](CLI_USAGE.md)
- **W&B Docs:** https://docs.wandb.ai/

---

**Implementation Date:** January 27, 2026
**W&B Project:** debates-doe
**Token:** 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 (configured)

Ready to run experiments! 🚀
