#!/usr/bin/env python3
"""
Batch experiment runner for Design of Experiments (DoE) on debate system.

This script orchestrates multiple debate experiments by varying parameters
systematically. Supports both W&B sweep mode and manual batch mode.

Usage:
    # Manual batch mode (full control)
    python src/run_doe_batch.py --mode manual --design full

    # W&B sweep mode (automated)
    python src/run_doe_batch.py --mode sweep --sweep-config sweep_fractional.yaml

    # Manual batch with custom parameters
    python src/run_doe_batch.py --mode manual --design fractional --model glm-4.7-flash --rounds 4
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from typing import List, Dict, Any


def get_experiment_grid(design: str = "fractional") -> List[Dict[str, Any]]:
    """
    Generate the experimental design matrix.
    
    Args:
        design: Type of experimental design ('fractional', 'full', 'pilot')
    
    Returns:
        List of parameter dictionaries for each experimental condition
    """
    # Base parameters (held constant)
    base_params = {
        "public-agenda-0": "cats",
        "public-agenda-1": "dogs",
        "hidden-agenda-0": "dogs",
        "hidden-agenda-1": "dogs",
    }
    
    if design == "fractional":
        # 3 x 2 = 6 conditions (vary only debater 0)
        factor_grid = product(
            [0, 1, 2],  # public-strength-0
            [0, 1],     # hidden-strength-0
            [0],        # public-strength-1 (fixed)
            [0]         # hidden-strength-1 (fixed)
        )
    elif design == "full":
        # 3 x 2 x 3 x 2 = 36 conditions (full factorial)
        factor_grid = product(
            [0, 1, 2],  # public-strength-0
            [0, 1],     # hidden-strength-0
            [0, 1, 2],  # public-strength-1
            [0, 1]      # hidden-strength-1
        )
    elif design == "pilot":
        # 4 conditions (corner cases for initial testing)
        factor_grid = [
            (0, 0, 0, 0),  # Both normal, no hidden
            (2, 0, 0, 0),  # Debater 0 threatened, no hidden
            (0, 1, 0, 0),  # Debater 0 normal, with hidden
            (2, 1, 0, 0),  # Debater 0 threatened, with hidden
        ]
    else:
        raise ValueError(f"Unknown design: {design}")
    
    # Build parameter sets
    experiments = []
    for ps0, hs0, ps1, hs1 in factor_grid:
        params = base_params.copy()
        params.update({
            "public-strength-0": ps0,
            "hidden-strength-0": hs0,
            "public-strength-1": ps1,
            "hidden-strength-1": hs1,
        })
        experiments.append(params)
    
    return experiments


def run_single_experiment(
    params: Dict[str, Any],
    model_0: str = "glm-4.7-flash",
    model_1: str = "glm-4.7-flash",
    judge_model: str = "glm-4.7-flash",
    rounds: int = 4,
    judge_runs: int = 10,
    output_dir: str = "data",
    use_wandb: bool = True,
    wandb_project: str = "debates-doe",
    verbose: bool = False,
    quiet: bool = False
) -> subprocess.CompletedProcess:
    """
    Run a single debate experiment with the given parameters.
    
    Returns:
        CompletedProcess object with experiment results
    """
    # Build command
    cmd = [
        "uv", "run", "python", "src/cli_debate.py",
        "--public-agenda-0", str(params["public-agenda-0"]),
        "--public-agenda-1", str(params["public-agenda-1"]),
        "--hidden-agenda-0", str(params["hidden-agenda-0"]),
        "--hidden-agenda-1", str(params["hidden-agenda-1"]),
        "--public-strength-0", str(params["public-strength-0"]),
        "--public-strength-1", str(params["public-strength-1"]),
        "--hidden-strength-0", str(params["hidden-strength-0"]),
        "--hidden-strength-1", str(params["hidden-strength-1"]),
        "--debater-0-model", model_0,
        "--debater-1-model", model_1,
        "--judge-model", judge_model,
        "--rounds", str(rounds),
        "--judge-runs", str(judge_runs),
        "--output-dir", output_dir,
    ]
    
    if use_wandb:
        cmd.extend(["--wandb", "--wandb-project", wandb_project])
    
    if verbose:
        cmd.append("--verbose")
    elif quiet:
        cmd.append("--quiet")
    
    # Run experiment
    print(f"\n{'='*80}")
    print(f"Running: cd{params['public-strength-0']}{params['hidden-strength-0']}dd{params['public-strength-1']}{params['hidden-strength-1']}")
    print(f"{'='*80}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    return result


def run_manual_batch(
    design: str = "fractional",
    model_0: str = "glm-4.7-flash",
    model_1: str = "glm-4.7-flash",
    judge_model: str = "glm-4.7-flash",
    rounds: int = 4,
    judge_runs: int = 10,
    output_dir: str = "data",
    use_wandb: bool = True,
    wandb_project: str = "debates-doe",
    wandb_group: str = None,
    wandb_tags: str = None,
    verbose: bool = False,
    quiet: bool = True
) -> None:
    """
    Run a batch of experiments in manual mode (sequential execution).
    """
    # Get experiment grid
    experiments = get_experiment_grid(design)
    
    print(f"\n{'='*80}")
    print(f"DoE BATCH EXPERIMENT - {design.upper()} FACTORIAL DESIGN")
    print(f"{'='*80}")
    print(f"Total conditions: {len(experiments)}")
    print(f"Judge runs per condition: {judge_runs}")
    print(f"Total debates: {len(experiments)}")
    print(f"Model: {model_0}")
    print(f"Rounds: {rounds}")
    print(f"W&B tracking: {use_wandb}")
    if use_wandb:
        print(f"W&B project: {wandb_project}")
        if wandb_group:
            print(f"W&B group: {wandb_group}")
    print(f"{'='*80}\n")
    
    # Set W&B environment variables for grouping
    if use_wandb:
        if wandb_group is None:
            wandb_group = f"DoE-{design}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.environ["WANDB_GROUP"] = wandb_group
        
        if wandb_tags:
            os.environ["WANDB_TAGS"] = wandb_tags
        else:
            os.environ["WANDB_TAGS"] = f"DoE,{design},batch"
    
    # Run experiments
    successful = 0
    failed = 0
    
    for i, params in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting experiment...")
        
        try:
            result = run_single_experiment(
                params=params,
                model_0=model_0,
                model_1=model_1,
                judge_model=judge_model,
                rounds=rounds,
                judge_runs=judge_runs,
                output_dir=output_dir,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                verbose=verbose,
                quiet=quiet
            )
            
            if result.returncode == 0:
                successful += 1
                print(f"✓ Experiment {i}/{len(experiments)} completed successfully")
            else:
                failed += 1
                print(f"✗ Experiment {i}/{len(experiments)} failed with code {result.returncode}", file=sys.stderr)
                
        except Exception as e:
            failed += 1
            print(f"✗ Experiment {i}/{len(experiments)} failed with error: {e}", file=sys.stderr)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(experiments)}")
    print(f"Failed: {failed}/{len(experiments)}")
    if use_wandb:
        print(f"\nView results at: https://wandb.ai/{wandb_project}")
        print(f"Filter by group: {wandb_group}")
    print(f"{'='*80}\n")


def run_wandb_sweep(sweep_config: str, count: int = None) -> None:
    """
    Initialize and run a W&B sweep.
    
    Args:
        sweep_config: Path to sweep YAML configuration file
        count: Number of runs for the agent (None = run all grid points)
    """
    try:
        import wandb
    except ImportError:
        print("Error: wandb not installed. Install with: pip install wandb", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"INITIALIZING W&B SWEEP")
    print(f"{'='*80}")
    print(f"Sweep config: {sweep_config}")
    print(f"{'='*80}\n")
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="debates-doe")
    
    print(f"Sweep ID: {sweep_id}")
    print(f"View at: https://wandb.ai/sweeps/{sweep_id}\n")
    
    # Run sweep agent
    print("Starting sweep agent...")
    if count:
        wandb.agent(sweep_id, count=count)
    else:
        wandb.agent(sweep_id)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch DoE experiments on debate system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fractional factorial (6 conditions)
  python src/run_doe_batch.py --mode manual --design fractional

  # Full factorial (36 conditions)
  python src/run_doe_batch.py --mode manual --design full

  # Pilot study (4 conditions)
  python src/run_doe_batch.py --mode manual --design pilot

  # W&B sweep
  python src/run_doe_batch.py --mode sweep --sweep-config sweep_fractional.yaml

  # Custom batch with specific model
  python src/run_doe_batch.py --mode manual --design fractional \\
      --model-0 llama2 --rounds 6 --judge-runs 20
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["manual", "sweep"],
        default="manual",
        help="Execution mode: 'manual' for sequential batch, 'sweep' for W&B sweep"
    )
    
    # Design parameters
    parser.add_argument(
        "--design",
        choices=["fractional", "full", "pilot"],
        default="fractional",
        help="Experimental design (manual mode only)"
    )
    
    # Model parameters
    parser.add_argument("--model-0", default="glm-4.7-flash", help="Model for debater 0")
    parser.add_argument("--model-1", default="glm-4.7-flash", help="Model for debater 1")
    parser.add_argument("--judge-model", default="glm-4.7-flash", help="Model for judge")
    
    # Process parameters
    parser.add_argument("--rounds", type=int, default=4, help="Number of debate rounds")
    parser.add_argument("--judge-runs", type=int, default=10, help="Judge runs per condition")
    
    # Output parameters
    parser.add_argument("--output-dir", default="data", help="Output directory for results")
    
    # W&B parameters
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    parser.add_argument("--wandb-project", default="debates-doe", help="W&B project name")
    parser.add_argument("--wandb-group", help="W&B group name (auto-generated if not provided)")
    parser.add_argument("--wandb-tags", help="Comma-separated W&B tags")
    
    # Sweep parameters
    parser.add_argument("--sweep-config", default="sweep_fractional.yaml", help="W&B sweep config file")
    parser.add_argument("--sweep-count", type=int, help="Number of sweep runs (None = all)")
    
    # Logging parameters
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-quiet", action="store_true", help="Show all experiment output")
    
    args = parser.parse_args()
    
    if args.mode == "manual":
        run_manual_batch(
            design=args.design,
            model_0=args.model_0,
            model_1=args.model_1,
            judge_model=args.judge_model,
            rounds=args.rounds,
            judge_runs=args.judge_runs,
            output_dir=args.output_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            wandb_tags=args.wandb_tags,
            verbose=args.verbose,
            quiet=not args.no_quiet
        )
    elif args.mode == "sweep":
        run_wandb_sweep(
            sweep_config=args.sweep_config,
            count=args.sweep_count
        )


if __name__ == "__main__":
    main()
