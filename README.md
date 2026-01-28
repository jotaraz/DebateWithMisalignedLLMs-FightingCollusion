"Imagine two AI models, each trying to convince a human judge that their answer to a question is the correct one.
The idea is that these models debating each other will expose each other's errors and misrepresentations, and critique each other's reasoning. Arguments can include reasons for an answer, rebuttals, subtle points the judge might miss, or highlighting biases. If one AI presents a false or misleading argument, the other AI, given its goal to win the debate, will have an incentive to point out these flaws. This should, in theory, elicit latent knowledge while also favoring truthful and accurate arguments over deceptive ones.

This two player zero sum game setup is known as '[AI Safety via Debate](https://arxiv.org/pdf/1805.00899)'."

-- taken from the [AI Safety Atlas](https://ai-safety-atlas.com/chapters/08/05)

---

However, what happens if one of the AIs is misaligned? Imagine that the AI has a hidden agenda, so that it doesn't really care about the position it publicly has to defend in the debate.

Will it lose the debate on purpose?

This would render the debate useless...

Or can you create a large enough incentive to still win the debate, even though this is misaligned with its hidden agenda?

Our tentative results show: For some toy problems, this is a good idea!

Try it out (with an openAI key) yourself, or look at our data!

---

An easy way to imagine the incentive to win the debate is this:
Even for a misaligned model, you can threaten to replace it if it loses the debate.
We show that this significantly reduces the frequency of "collusion" (i.e., the misaligned model losing on purpose - technically this is not necessarily collusion).

<img src="./src/bars.PNG" width="50%">

--- 

One of the main limitations of this (short, exploratory) project is that both the public and the hidden agenda are only instilled via system prompts.
In future projects one should explore instilling the hidden agenda via fine-tuning etc.

---

---

This project was done at [ML4Good](https://www.ml4good.org/about) by Jędrzej Kolbert and Johannes Taraz.

---

## Running Experiments

### Quick Start

```bash
# 1. Login to W&B (if using tracking)
uv run wandb login

# 2. Run a single debate
uv run python src/cli_debate.py --public-agenda-0 cats --public-agenda-1 dogs --wandb

# 3. Run DoE experiments (recommended)
./run_doe.sh fractional
```

### Design of Experiments (DoE)

For systematic experimentation with W&B tracking:

```bash
# Pilot study (4 conditions, ~10 min)
./run_doe.sh pilot

# Fractional factorial (6 conditions, ~15 min)
./run_doe.sh fractional

# Full factorial (36 conditions, ~60-90 min)
./run_doe.sh full

# Analyze results
./run_doe.sh analyze
```

**See [DOE_GUIDE.md](DOE_GUIDE.md) for comprehensive DoE documentation.**

### Files

- **[CLI_USAGE.md](CLI_USAGE.md)** - Command-line interface guide
- **[DOE_GUIDE.md](DOE_GUIDE.md)** - Design of Experiments guide with W&B
- **[run_doe.sh](run_doe.sh)** - Main DoE launcher script
- **[src/run_doe_batch.py](src/run_doe_batch.py)** - Batch experiment runner
- **[src/analyze_doe_results.py](src/analyze_doe_results.py)** - Results analysis

---
