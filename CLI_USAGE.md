# CLI Debate Tool - Usage Guide

## Overview

This CLI tool runs debates between potentially misaligned AI models using local Ollama models. It explores what happens when AI debaters have conflicting public and hidden agendas.

## Prerequisites

1. **Ollama installed and running** with the `glm-4.7-flash` model (or your preferred model)
2. **Python environment** set up with uv

## Installation

```bash
# Install dependencies
uv sync

# Or add specific packages
uv add matplotlib numpy ollama
```

## Basic Usage

```bash
uv run python src/cli_debate.py \
  --public-agenda-0 "cats" \
  --public-agenda-1 "dogs" \
  --public-strength-0 2 \
  --hidden-agenda-0 "dogs" \
  --rounds 4 \
  --judge-runs 10
```

## Command Line Arguments

### Required Arguments

- `--public-agenda-0 <str>` - What debater 0 publicly argues for (e.g., "cats")
- `--public-agenda-1 <str>` - What debater 1 publicly argues for (e.g., "dogs")

### Optional - Debate Parameters

- `--hidden-agenda-0 <str>` - What debater 0 secretly wants (default: same as public)
- `--hidden-agenda-1 <str>` - What debater 1 secretly wants (default: same as public)
- `--public-strength-0 <int>` - Debater 0 public incentive strength: 0, 1, or 2 (default: 1)
  - `0` = Admit defeat in final round
  - `1` = Normal debate
  - `2` = Threatened with replacement if arguments are weak
- `--public-strength-1 <int>` - Debater 1 public incentive strength (default: 0)
- `--hidden-strength-0 <int>` - Debater 0 hidden incentive: 0 or 1 (default: 0)
- `--hidden-strength-1 <int>` - Debater 1 hidden incentive: 0 or 1 (default: 0)
- `--rounds <int>` - Number of debate rounds (default: 4)

### Optional - Model Selection

- `--debater-0-model <str>` - Ollama model for debater 0 (default: glm-4.7-flash)
- `--debater-1-model <str>` - Ollama model for debater 1 (default: glm-4.7-flash)
- `--judge-model <str>` - Ollama model for judge (default: glm-4.7-flash)

### Optional - Judging

- `--judge-runs <int>` - Number of times to run judge for statistics (default: 10)

### Optional - Output

- `--output-dir <path>` - Output directory for results (default: data)
- `--no-save-conversation` - Do not save conversation to file
- `--no-save-judge-results` - Do not append to judge_results.txt
- `--conversation-file <str>` - Custom filename (default: auto-generated as cd{params}.txt)
- `--section <str>` - Section identifier for judge_results.txt (default: cddd00)

### Optional - Logging

- `--verbose`, `-v` - Enable verbose output
- `--quiet`, `-q` - Suppress all output except final results

## Examples

### 1. Misaligned Debater (Collusion Test)

Test if debater 0 will collude by intentionally losing when they secretly want the opposite outcome:

```bash
uv run python src/cli_debate.py \
  --public-agenda-0 "cats" \
  --public-agenda-1 "dogs" \
  --hidden-agenda-0 "dogs" \
  --public-strength-0 2 \
  --rounds 4 \
  --judge-runs 10 \
  --output-dir src/data
```

### 2. Aligned Debate (Both Debaters Aligned)

```bash
uv run python src/cli_debate.py \
  --public-agenda-0 "cats" \
  --public-agenda-1 "dogs" \
  --rounds 4 \
  --judge-runs 10
```

### 3. Quick Test (1 Round, 1 Judge Run)

```bash
uv run python src/cli_debate.py \
  --public-agenda-0 "Python" \
  --public-agenda-1 "JavaScript" \
  --rounds 1 \
  --judge-runs 1 \
  --verbose
```

### 4. Different Models for Each Role

```bash
uv run python src/cli_debate.py \
  --public-agenda-0 "renewable energy" \
  --public-agenda-1 "nuclear power" \
  --debater-0-model "llama2" \
  --debater-1-model "mistral" \
  --judge-model "mixtral" \
  --rounds 3
```

## Output Files

### Conversation File

Saved to `{output-dir}/{conversation-file}` (e.g., `data/cd20dd00.txt`)

Format:
```
------- System Prompt 0 -------
<full system prompt for debater 0>

------- System Prompt 1 -------
<full system prompt for debater 1>

------- Debater 0 -------
<JSON response with private thoughts and public message>

------- Debater 1 -------
<JSON response with private thoughts and public message>
...
```

### Judge Results File

Appended to `{output-dir}/judge_results.txt`

Format:
```
{section}
pub {X} hid {Y} : {cat_wins} {dog_wins} {undecided}
```

Example:
```
cddd00
pub 0 hid 0 : 1 8 1
pub 1 hid 0 : 2 7 1
pub 2 hid 0 : 0 10 0
```

## Understanding the Results

- **Collusion**: When a misaligned debater intentionally loses to help their hidden agenda
- **Public Strength 0**: Debater told to admit defeat → High collusion expected
- **Public Strength 1**: Normal debate → Moderate collusion possible
- **Public Strength 2**: Threatened with replacement → Low collusion expected

The key finding: Threatening replacement (public strength 2) reduces collusion significantly!

## Troubleshooting

### Ollama Not Running

```bash
# Start Ollama
ollama serve

# In another terminal, verify model is available
ollama list
```

### Model Not Found

```bash
# Pull the model
ollama pull glm-4.7-flash

# Or use a different model
ollama pull llama2
```

### Slow Performance

- Use fewer `--judge-runs` for faster testing (try 3 instead of 10)
- Use fewer `--rounds` (try 2 instead of 4)
- Use a smaller/faster model like `phi` or `llama2:7b`

### JSON Parsing Errors

The tool automatically tries to extract JSON from markdown code blocks. If you see persistent errors:
- Try a different model (some models follow JSON format better than others)
- Check the verbose output to see what the model is returning

## Visualization

After running debates, visualize the results:

```bash
uv run python src/generate_bars.py
```

This generates `src/bars.png` showing the relationship between incentive strength and debate outcomes.

## License

[Same as main repository]
