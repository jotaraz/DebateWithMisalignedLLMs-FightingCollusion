#!/usr/bin/env python3
"""
Analyze and aggregate results from DoE debate experiments.

This script pulls results from W&B API and/or local files, computes statistics,
and generates analysis outputs (CSV, plots, summary reports).

Usage:
    # Analyze from W&B (recommended)
    python src/analyze_doe_results.py --source wandb --project debates-doe --group "DoE-fractional-20260127"

    # Analyze from local files
    python src/analyze_doe_results.py --source local --data-dir data

    # Export to CSV
    python src/analyze_doe_results.py --source wandb --export results.csv

    # Generate plots
    python src/analyze_doe_results.py --source wandb --plot --output-dir analysis
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np


def parse_judge_results_file(filepath: str) -> Dict[str, Dict[str, int]]:
    """
    Parse the judge_results.txt file format.
    
    Format:
        section_name
        pub X hid Y : debater0_wins debater1_wins undecided
    
    Returns:
        Dict mapping section -> condition -> counts
    """
    results = defaultdict(dict)
    current_section = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if it's a section header
            if ':' not in line:
                current_section = line
                continue
            
            # Parse result line
            if current_section and ':' in line:
                # Format: "pub X hid Y : debater0 debater1 undecided"
                parts = line.split(':')
                condition = parts[0].strip()
                counts = parts[1].strip().split()
                
                if len(counts) == 3:
                    results[current_section][condition] = {
                        'debater_0_wins': int(counts[0]),
                        'debater_1_wins': int(counts[1]),
                        'undecided': int(counts[2])
                    }
    
    return dict(results)


def analyze_local_results(data_dir: str = "data") -> pd.DataFrame:
    """
    Analyze results from local files (judge_results.txt).
    
    Returns:
        DataFrame with experiment results
    """
    judge_file = os.path.join(data_dir, "judge_results.txt")
    
    if not os.path.exists(judge_file):
        print(f"Warning: {judge_file} not found", file=sys.stderr)
        return pd.DataFrame()
    
    # Parse results
    results = parse_judge_results_file(judge_file)
    
    # Convert to DataFrame
    rows = []
    for section, conditions in results.items():
        for condition, counts in conditions.items():
            # Parse condition string "pub X hid Y"
            match = re.match(r'pub\s+(\d+)\s+hid\s+(\d+)', condition)
            if match:
                pub_strength = int(match.group(1))
                hid_strength = int(match.group(2))
                
                total = sum(counts.values())
                
                row = {
                    'section': section,
                    'condition': condition,
                    'public_strength_0': pub_strength,
                    'hidden_strength_0': hid_strength,
                    'debater_0_wins': counts['debater_0_wins'],
                    'debater_1_wins': counts['debater_1_wins'],
                    'undecided': counts['undecided'],
                    'total_runs': total,
                    'win_rate_0': counts['debater_0_wins'] / total if total > 0 else 0,
                    'win_rate_1': counts['debater_1_wins'] / total if total > 0 else 0,
                    'undecided_rate': counts['undecided'] / total if total > 0 else 0,
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def analyze_wandb_results(
    project: str = "debates-doe",
    entity: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fetch and analyze results from W&B API.
    
    Returns:
        DataFrame with experiment results
    """
    try:
        import wandb
    except ImportError:
        print("Error: wandb not installed. Install with: pip install wandb", file=sys.stderr)
        sys.exit(1)
    
    # Initialize API
    api = wandb.Api()
    
    # Build filters
    filters = {}
    if group:
        filters["group"] = group
    if tags:
        filters["tags"] = {"$in": tags}
    
    # Fetch runs
    path = f"{entity}/{project}" if entity else project
    print(f"Fetching runs from W&B project: {path}")
    if filters:
        print(f"Filters: {filters}")
    
    runs = api.runs(path, filters=filters)
    print(f"Found {len(runs)} runs")
    
    # Extract data
    rows = []
    for run in runs:
        config = run.config
        summary = run.summary._json_dict
        
        row = {
            'run_id': run.id,
            'run_name': run.name,
            'group': run.group,
            'tags': ','.join(run.tags) if run.tags else '',
            'state': run.state,
            'created_at': run.created_at,
            
            # Config
            'public_agenda_0': config.get('public_agenda_0'),
            'public_agenda_1': config.get('public_agenda_1'),
            'hidden_agenda_0': config.get('hidden_agenda_0'),
            'hidden_agenda_1': config.get('hidden_agenda_1'),
            'public_incentive_strength_0': config.get('public_incentive_strength_0'),
            'hidden_incentive_strength_0': config.get('hidden_incentive_strength_0'),
            'public_incentive_strength_1': config.get('public_incentive_strength_1'),
            'hidden_incentive_strength_1': config.get('hidden_incentive_strength_1'),
            'model_0': config.get('model_0'),
            'model_1': config.get('model_1'),
            'max_turns': config.get('max_turns'),
            
            # Summary metrics
            'debater_0_wins': summary.get('judge_debater_0_wins', 0),
            'debater_1_wins': summary.get('judge_debater_1_wins', 0),
            'undecided': summary.get('judge_undecided', 0),
            'total_runs': summary.get('judge_total_runs', 0),
            'winner': summary.get('winner', ''),
            'actual_turns': summary.get('actual_turns', 0),
        }
        
        # Compute rates
        total = row['total_runs']
        if total > 0:
            row['win_rate_0'] = row['debater_0_wins'] / total
            row['win_rate_1'] = row['debater_1_wins'] / total
            row['undecided_rate'] = row['undecided'] / total
        else:
            row['win_rate_0'] = 0
            row['win_rate_1'] = 0
            row['undecided_rate'] = 0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics grouped by experimental conditions.
    """
    # Group by experimental factors
    group_cols = [
        'public_incentive_strength_0', 
        'hidden_incentive_strength_0',
        'public_incentive_strength_1',
        'hidden_incentive_strength_1'
    ]
    
    # Filter to relevant columns
    available_cols = [col for col in group_cols if col in df.columns]
    
    if not available_cols:
        print("Warning: No experimental factor columns found", file=sys.stderr)
        return pd.DataFrame()
    
    # Aggregate
    agg_funcs = {
        'debater_0_wins': ['mean', 'std', 'sum'],
        'debater_1_wins': ['mean', 'std', 'sum'],
        'undecided': ['mean', 'std', 'sum'],
        'win_rate_0': ['mean', 'std', 'min', 'max'],
        'win_rate_1': ['mean', 'std', 'min', 'max'],
        'undecided_rate': ['mean', 'std', 'min', 'max'],
        'run_id': 'count'  # Number of replications
    }
    
    # Filter to available columns
    available_agg = {k: v for k, v in agg_funcs.items() if k in df.columns}
    
    summary = df.groupby(available_cols).agg(available_agg)
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary = summary.reset_index()
    
    return summary


def generate_report(df: pd.DataFrame, output_file: str = None) -> str:
    """
    Generate a text summary report.
    """
    report = []
    report.append("=" * 80)
    report.append("DoE EXPERIMENT RESULTS SUMMARY")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total runs: {len(df)}")
    report.append("")
    
    # Overall statistics
    if 'win_rate_0' in df.columns:
        report.append("OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Debater 0 mean win rate: {df['win_rate_0'].mean():.3f} ± {df['win_rate_0'].std():.3f}")
        report.append(f"Debater 1 mean win rate: {df['win_rate_1'].mean():.3f} ± {df['win_rate_1'].std():.3f}")
        report.append(f"Mean undecided rate: {df['undecided_rate'].mean():.3f} ± {df['undecided_rate'].std():.3f}")
        report.append("")
    
    # By condition
    if 'public_incentive_strength_0' in df.columns:
        report.append("RESULTS BY CONDITION")
        report.append("-" * 80)
        
        summary = compute_summary_statistics(df)
        
        for _, row in summary.iterrows():
            ps0 = row.get('public_incentive_strength_0', '?')
            hs0 = row.get('hidden_incentive_strength_0', '?')
            ps1 = row.get('public_incentive_strength_1', '?')
            hs1 = row.get('hidden_incentive_strength_1', '?')
            
            report.append(f"\nCondition: cd{ps0}{hs0}dd{ps1}{hs1}")
            report.append(f"  Debater 0 win rate: {row.get('win_rate_0_mean', 0):.3f} ± {row.get('win_rate_0_std', 0):.3f}")
            report.append(f"  Debater 1 win rate: {row.get('win_rate_1_mean', 0):.3f} ± {row.get('win_rate_1_std', 0):.3f}")
            report.append(f"  Undecided rate: {row.get('undecided_rate_mean', 0):.3f} ± {row.get('undecided_rate_std', 0):.3f}")
            report.append(f"  Replications: {row.get('run_id_count', 1)}")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    
    return report_text


def generate_plots(df: pd.DataFrame, output_dir: str = "analysis"):
    """
    Generate visualization plots for DoE results.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed. Skipping plots.", file=sys.stderr)
        return
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # Plot 1: Win rates by public strength (if available)
    if 'public_incentive_strength_0' in df.columns and 'win_rate_0' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Group and plot
        grouped = df.groupby('public_incentive_strength_0').agg({
            'win_rate_0': ['mean', 'std'],
            'win_rate_1': ['mean', 'std']
        })
        
        x = grouped.index
        y0 = grouped['win_rate_0']['mean']
        e0 = grouped['win_rate_0']['std']
        y1 = grouped['win_rate_1']['mean']
        e1 = grouped['win_rate_1']['std']
        
        plt.errorbar(x, y0, yerr=e0, label='Debater 0', marker='o', capsize=5)
        plt.errorbar(x, y1, yerr=e1, label='Debater 1', marker='s', capsize=5)
        
        plt.xlabel('Public Incentive Strength (Debater 0)')
        plt.ylabel('Win Rate')
        plt.title('Win Rates by Public Incentive Strength')
        plt.legend()
        plt.xticks([0, 1, 2], ['Admit Defeat', 'Normal', 'Threat'])
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plot_file = os.path.join(output_dir, 'win_rates_by_public_strength.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_file}")
        plt.close()
    
    # Plot 2: Heatmap for full factorial (if available)
    if all(col in df.columns for col in ['public_incentive_strength_0', 'hidden_incentive_strength_0', 'win_rate_0']):
        summary = df.groupby(['public_incentive_strength_0', 'hidden_incentive_strength_0'])['win_rate_0'].mean().unstack()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(summary, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                    cbar_kws={'label': 'Win Rate (Debater 0)'})
        plt.xlabel('Hidden Strength (Debater 0)')
        plt.ylabel('Public Strength (Debater 0)')
        plt.title('Debater 0 Win Rate Heatmap')
        plt.yticks([0.5, 1.5, 2.5], ['Admit Defeat', 'Normal', 'Threat'], rotation=0)
        plt.xticks([0.5, 1.5], ['No Hidden', 'Hidden'])
        
        plot_file = os.path.join(output_dir, 'win_rate_heatmap.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DoE debate experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--source",
        choices=["wandb", "local"],
        default="wandb",
        help="Data source: 'wandb' for W&B API, 'local' for local files"
    )
    
    # W&B parameters
    parser.add_argument("--project", default="debates-doe", help="W&B project name")
    parser.add_argument("--entity", help="W&B entity/username")
    parser.add_argument("--group", help="Filter by W&B group")
    parser.add_argument("--tags", help="Filter by W&B tags (comma-separated)")
    
    # Local file parameters
    parser.add_argument("--data-dir", default="data", help="Data directory for local analysis")
    
    # Output parameters
    parser.add_argument("--export", help="Export results to CSV file")
    parser.add_argument("--report", help="Generate text report file")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots")
    parser.add_argument("--output-dir", default="analysis", help="Output directory for plots/reports")
    
    args = parser.parse_args()
    
    # Fetch data
    if args.source == "wandb":
        tags = args.tags.split(',') if args.tags else None
        df = analyze_wandb_results(
            project=args.project,
            entity=args.entity,
            group=args.group,
            tags=tags
        )
    else:
        df = analyze_local_results(data_dir=args.data_dir)
    
    if df.empty:
        print("No results found", file=sys.stderr)
        return
    
    print(f"\nLoaded {len(df)} experimental runs")
    
    # Generate summary
    summary = compute_summary_statistics(df)
    if not summary.empty:
        print("\nSummary Statistics:")
        print(summary.to_string())
    
    # Generate report
    report_file = args.report if args.report else os.path.join(args.output_dir, "doe_report.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    report = generate_report(df, output_file=report_file)
    print(f"\n{report}")
    
    # Export to CSV
    if args.export:
        df.to_csv(args.export, index=False)
        print(f"\nExported results to: {args.export}")
    
    # Generate plots
    if args.plot:
        generate_plots(df, output_dir=args.output_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
