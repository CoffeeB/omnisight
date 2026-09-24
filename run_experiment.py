#!/usr/bin/env python3
"""
OmniSight: Research Experiment CLI Runner
Unified entry point for executing reproducibility benchmarks across all 5 experimental protocols.
"""

import argparse
import sys
import os
import json
from typing import Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from experiments.exp01_viewpoint import run_viewpoint_experiment
from experiments.exp02_altitude import run_altitude_experiment
from experiments.exp03_occlusion import run_occlusion_experiment
from experiments.exp04_environmental import run_environmental_experiment
from experiments.exp05_calibration import run_calibration_experiment
from src.visualization.dashboard import generate_research_dashboard

console = Console()

def print_banner():
    banner_text = """
 [bold cyan]OMNISIGHT[/bold cyan] — Research Benchmark Platform
 Multi-View Semantic Terrain & Environment Understanding
 Laboratory for Robust Spatial Intelligence
    """
    console.print(Panel(banner_text, style="blue", expand=False))


def display_results_table(exp_name: str, results: Dict[str, Any]):
    table = Table(title=f"Results: {exp_name}", show_header=True, header_style="bold magenta")
    table.add_column("Key / Condition", style="cyan", width=24)
    table.add_column("Value / Metric", style="green", width=36)

    for k, v in results.items():
        if isinstance(v, dict):
            sub_str = json.dumps(v, indent=2)
            table.add_row(k, sub_str)
        else:
            table.add_row(k, str(v))

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="OmniSight Multi-View & Atmospheric Robustness Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default="all",
        choices=["viewpoint", "altitude", "occlusion", "environmental", "calibration", "all"],
        help="Experiment identifier to run"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=25,
        help="Number of synthetic/dataset scene samples to evaluate per condition"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./results_benchmark",
        help="Output directory for results JSON and visual figures"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Render and save publication figures and reliability diagrams"
    )

    args = parser.parse_args()
    print_banner()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    exp_choice = args.experiment.lower()

    if exp_choice in ["viewpoint", "all"]:
        console.print("[bold yellow]Executing Experiment 01: Viewpoint Robustness...[/bold yellow]")
        exp01_out = os.path.join(args.output_dir, "exp01_viewpoint")
        res1 = run_viewpoint_experiment(num_samples=args.samples, output_dir=exp01_out, visualize=args.visualize)
        all_results["viewpoint"] = res1
        display_results_table("EXP-01 Viewpoint Invariance", res1.get("details", {}))

    if exp_choice in ["altitude", "all"]:
        console.print("[bold yellow]Executing Experiment 02: Altitude GSD Variation...[/bold yellow]")
        exp02_out = os.path.join(args.output_dir, "exp02_altitude")
        res2 = run_altitude_experiment(num_samples=args.samples, output_dir=exp02_out, visualize=args.visualize)
        all_results["altitude"] = res2.get("altitude_results", {})
        display_results_table("EXP-02 Altitude GSD Invariance", res2.get("details", {}))

    if exp_choice in ["occlusion", "all"]:
        console.print("[bold yellow]Executing Experiment 03: Structured Occlusion Degradation...[/bold yellow]")
        exp03_out = os.path.join(args.output_dir, "exp03_occlusion")
        res3 = run_occlusion_experiment(num_samples=args.samples, output_dir=exp03_out, visualize=args.visualize)
        all_results["occlusion"] = res3.get("occlusion_results", {})
        display_results_table("EXP-03 Occlusion Robustness", res3.get("details", {}))

    if exp_choice in ["environmental", "all"]:
        console.print("[bold yellow]Executing Experiment 04: Environmental & Atmospheric Scattering...[/bold yellow]")
        exp04_out = os.path.join(args.output_dir, "exp04_environmental")
        res4 = run_environmental_experiment(num_samples=args.samples, output_dir=exp04_out, visualize=args.visualize)
        all_results["environmental"] = res4.get("environmental_results", {})
        display_results_table("EXP-04 Environmental Attenuation", res4.get("details", {}))

    if exp_choice in ["calibration", "all"]:
        console.print("[bold yellow]Executing Experiment 05: Confidence Calibration & ECE...[/bold yellow]")
        exp05_out = os.path.join(args.output_dir, "exp05_calibration")
        res5 = run_calibration_experiment(num_samples=args.samples, output_dir=exp05_out, visualize=args.visualize)
        all_results["calibration"] = res5
        display_results_table("EXP-05 Confidence Calibration", res5)

    # Save comprehensive results JSON
    summary_json_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate interactive HTML dashboard
    dashboard_path = os.path.join(args.output_dir, "research_dashboard.html")
    generate_research_dashboard(all_results, dashboard_path)

    console.print(f"\n[bold green]✓ Benchmark run complete![/bold green]")
    console.print(f"Summary JSON: [cyan]{summary_json_path}[/cyan]")
    console.print(f"Interactive Dashboard: [cyan]{dashboard_path}[/cyan]")

if __name__ == "__main__":
    main()
