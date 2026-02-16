#!/usr/bin/env python3
"""
Statistical Tests for Transcription Post-Processing Methods Comparison
=======================================================================

This script performs statistical significance tests (Wilcoxon signed-rank test
and paired t-test) to compare different post-processing methods.

Demonstrates whether improvements between methods (e.g., single-pass vs two-pass,
two-pass vs two-pass-reversed) are statistically significant.

Usage:
    python src/evaluation/statistical_tests.py

    # Compare specific methods
    python src/evaluation/statistical_tests.py --methods baseline qwen_80b_twopass

    # Focus on specific metric
    python src/evaluation/statistical_tests.py --metric wder

    # Export results to a formatted table
    python src/evaluation/statistical_tests.py --table

Output:
    - Console output with p-values and significance indicators
    - JSON file with all test results
    - Optional formatted table for publication
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EVALUATIONS_DIR = PROJECT_ROOT / "results" / "evaluations"
DETAILED_RESULTS_DIR = EVALUATIONS_DIR / "detailed"
OUTPUT_DIR = EVALUATIONS_DIR / "statistical_tests"


@dataclass
class TestResult:
    """Result of a statistical test between two methods."""
    method_a: str
    method_b: str
    metric: str
    domain: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    difference: float  # mean_b - mean_a (negative = improvement)
    relative_improvement: float  # percentage improvement
    n_samples: int
    wilcoxon_statistic: Optional[float]
    wilcoxon_pvalue: Optional[float]
    significant_wilcoxon: Optional[bool]  # at alpha=0.05 (uncorrected)


def load_detailed_results(method: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load detailed per-file results for a method.

    Returns:
        Dict with structure: {domain: {filename: {metric: value}}}
    """
    detailed_path = DETAILED_RESULTS_DIR / f"eval_detailed_{method}.json"

    if detailed_path.exists():
        with open(detailed_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: try to reconstruct from summary (less accurate)
    summary_path = EVALUATIONS_DIR / f"eval_summary_{method}.json"
    if not summary_path.exists():
        print(f"Warning: No results found for method '{method}'")
        return {}

    # Can't do paired tests without per-file data
    print(f"Warning: Only summary data available for '{method}'. "
          f"Run evaluate_results.py to generate per-file data.")
    return {}


def collect_all_detailed_results() -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Collect detailed results for all available methods.

    Returns:
        Dict with structure: {method: {domain: {filename: {metric: value}}}}
    """
    results = {}

    # Check for detailed results
    if DETAILED_RESULTS_DIR.exists():
        for f in DETAILED_RESULTS_DIR.glob("eval_detailed_*.json"):
            method = f.stem.replace("eval_detailed_", "")
            with open(f, "r", encoding="utf-8") as file:
                results[method] = json.load(file)

    return results


def extract_metric_values(
    detailed_data: Dict[str, Dict[str, float]],
    metric: str
) -> Tuple[List[str], List[float]]:
    """
    Extract metric values from detailed data, returning filenames and values.

    Args:
        detailed_data: {filename: {metric: value}}
        metric: metric name to extract

    Returns:
        Tuple of (filenames, values)
    """
    filenames = []
    values = []
    for filename, metrics in sorted(detailed_data.items()):
        if metric in metrics:
            filenames.append(filename)
            values.append(metrics[metric])
    return filenames, values


def get_available_methods() -> List[str]:
    """Get list of methods with available detailed results."""
    methods = []

    if DETAILED_RESULTS_DIR.exists():
        for f in DETAILED_RESULTS_DIR.glob("eval_detailed_*.json"):
            method = f.stem.replace("eval_detailed_", "")
            methods.append(method)

    # Also check for summary files (for listing purposes)
    for f in EVALUATIONS_DIR.glob("eval_summary_*.json"):
        method = f.stem.replace("eval_summary_", "")
        if method not in methods:
            methods.append(method)

    return sorted(methods)


def perform_statistical_tests(
    values_a: List[float],
    values_b: List[float],
    method_a: str,
    method_b: str,
    metric: str,
    domain: str,
    alpha: float = 0.05
) -> TestResult:
    """
    Perform Wilcoxon signed-rank test and paired t-test.

    The Wilcoxon test is non-parametric and doesn't assume normality.
    The t-test assumes normality but is more powerful when assumption holds.

    For small samples (n < 20), Wilcoxon is preferred.
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    n_samples = len(values_a)
    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    std_a = float(np.std(values_a))
    std_b = float(np.std(values_b))

    difference = mean_b - mean_a
    # Relative improvement: negative difference means improvement
    # (lower WER/DER/WDER is better)
    if mean_a > 0:
        relative_improvement = -difference / mean_a * 100
    else:
        relative_improvement = 0.0


    # Wilcoxon signed-rank test
    # Requires at least some non-zero differences
    differences = values_b - values_a
    if np.all(differences == 0):
        wilcoxon_stat = None
        wilcoxon_pval = None
        significant_wilcoxon = None
    else:
        try:
            # zero_method='wilcox' handles zeros by ranking them
            wilcoxon_stat, wilcoxon_pval = stats.wilcoxon(
                values_a, values_b,
                zero_method='wilcox',
                alternative='two-sided'
            )
            significant_wilcoxon = wilcoxon_pval < alpha
        except ValueError:
            # Can happen if all differences are zero
            wilcoxon_stat = None
            wilcoxon_pval = None
            significant_wilcoxon = None

    return TestResult(
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        domain=domain,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        difference=difference,
        relative_improvement=relative_improvement,
        n_samples=n_samples,
        wilcoxon_statistic=wilcoxon_stat,
        wilcoxon_pvalue=wilcoxon_pval,
        significant_wilcoxon=significant_wilcoxon,
    )


def is_file_outlier(file_data: Dict[str, float]) -> bool:
    """
    Check if a file is an outlier based on its metrics.

    A file is an outlier if any metric is > 100% or < 0%.
    """
    # Check if already marked as outlier
    if file_data.get("is_outlier", False):
        return True

    # Check metrics manually
    for metric in ["wer", "der", "wder"]:
        if metric in file_data:
            val = file_data[metric]
            if val > 1.0 or val < 0:
                return True
    return False


def compare_methods(
    all_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    method_a: str,
    method_b: str,
    metrics: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    exclude_outliers: bool = True
) -> List[TestResult]:
    """
    Compare two methods across all metrics and domains.

    Args:
        all_results: Detailed results for all methods
                     Structure: {method: {domain: {filename: {metric: value}}}}
        method_a: First method (baseline/reference)
        method_b: Second method (comparison)
        metrics: List of metrics to compare (default: all)
        domains: List of domains to compare (default: all)
        exclude_outliers: If True, exclude files with metrics > 100% or < 0%

    Returns:
        List of TestResult objects
    """
    if method_a not in all_results or method_b not in all_results:
        missing = []
        if method_a not in all_results:
            missing.append(method_a)
        if method_b not in all_results:
            missing.append(method_b)
        print(f"Error: Missing detailed results for: {', '.join(missing)}")
        return []

    results_a = all_results[method_a]
    results_b = all_results[method_b]

    if domains is None:
        domains = list(set(results_a.keys()) & set(results_b.keys()))

    if metrics is None:
        # Metrics relevant for evaluation
        metrics = ["wer", "der", "wder"]

    test_results = []

    for domain in domains:
        if domain not in results_a or domain not in results_b:
            continue

        # Get common files between methods
        files_a = set(results_a[domain].keys())
        files_b = set(results_b[domain].keys())
        common_files = sorted(files_a & files_b)

        # Filter out outliers if requested
        if exclude_outliers:
            valid_files = []
            for filename in common_files:
                is_outlier_a = is_file_outlier(results_a[domain][filename])
                is_outlier_b = is_file_outlier(results_b[domain][filename])
                if not is_outlier_a and not is_outlier_b:
                    valid_files.append(filename)
            if len(common_files) != len(valid_files):
                n_excluded = len(common_files) - len(valid_files)
                print(f"  [{domain}] Excluded {n_excluded} outlier(s), keeping {len(valid_files)} files")
            common_files = valid_files

        if len(common_files) < 3:
            print(f"Warning: Too few common files for {domain}: {len(common_files)}")
            continue

        for metric in metrics:
            # Extract values for common files (paired data)
            values_a = []
            values_b = []

            for filename in common_files:
                if metric in results_a[domain][filename] and metric in results_b[domain][filename]:
                    values_a.append(results_a[domain][filename][metric])
                    values_b.append(results_b[domain][filename][metric])

            if len(values_a) < 3:
                print(f"Warning: Too few samples for {domain}/{metric}: {len(values_a)}")
                continue

            result = perform_statistical_tests(
                values_a, values_b,
                method_a, method_b,
                metric, domain
            )
            test_results.append(result)

    return test_results


def format_pvalue(pval: Optional[float]) -> str:
    """Format p-value for display."""
    if pval is None:
        return "N/A"
    if pval < 0.001:
        return "<0.001***"
    elif pval < 0.01:
        return f"{pval:.3f}**"
    elif pval < 0.05:
        return f"{pval:.3f}*"
    else:
        return f"{pval:.3f}"


def print_results(test_results: List[TestResult], verbose: bool = True):
    """Print test results in a formatted table."""
    if not test_results:
        print("No test results to display.")
        return

    # Group by domain
    by_domain = defaultdict(list)
    for r in test_results:
        by_domain[r.domain].append(r)

    method_a = test_results[0].method_a
    method_b = test_results[0].method_b

    print("\n" + "=" * 80)
    print(f"STATISTICAL COMPARISON: {method_a} vs {method_b}")
    print("=" * 80)

    for domain, results in by_domain.items():
        print(f"\n{'─' * 80}")
        print(f"Domain: {domain.upper()}")
        print(f"{'─' * 80}")

        # Header
        print(f"{'Metric':<10} {'Mean A':>10} {'Mean B':>10} {'Δ':>10} "
              f"{'Δ%':>8} {'Wilcoxon':>12} {'t-test':>12} {'n':>4}")
        print("-" * 80)

        for r in sorted(results, key=lambda x: x.metric):
            delta_str = f"{r.difference:+.4f}"
            pct_str = f"{r.relative_improvement:+.1f}%"

            wilcox_str = format_pvalue(r.wilcoxon_pvalue)

            print(f"{r.metric:<10} {r.mean_a:>10.4f} {r.mean_b:>10.4f} "
                  f"{delta_str:>10} {pct_str:>8} {wilcox_str:>12} {r.n_samples:>4}")

        if verbose:
            print()
            print("Legend: * p<0.05, ** p<0.01, *** p<0.001")
            print("Δ% > 0 means improvement (lower error rate)")


def generate_table(
    test_results: List[TestResult],
    caption: str = "Statistical comparison of methods"
) -> str:
    """Generate a formatted table."""
    if not test_results:
        return ""

    method_a = test_results[0].method_a
    method_b = test_results[0].method_b

    # Group by domain
    by_domain = defaultdict(list)
    for r in test_results:
        by_domain[r.domain].append(r)

    table = []
    table.append("\\begin{table}[htbp]")
    table.append("\\centering")
    table.append(f"\\caption{{{caption}}}")
    table.append("\\label{tab:statistical_tests}")
    table.append("\\begin{tabular}{llrrrrr}")
    table.append("\\toprule")
    table.append("Domain & Metric & Mean A & Mean B & $\\Delta$\\% & Wilcoxon $p$ & Sig. \\\\")
    table.append("\\midrule")

    for domain, results in by_domain.items():
        domain_label = domain.replace("_", "\\_")
        for i, r in enumerate(sorted(results, key=lambda x: x.metric)):
            d = domain_label if i == 0 else ""
            metric = r.metric.upper()
            sig = "\\checkmark" if r.significant_wilcoxon else ""
            pval = f"{r.wilcoxon_pvalue:.3f}" if r.wilcoxon_pvalue else "N/A"

            table.append(
                f"{d} & {metric} & {r.mean_a:.3f} & {r.mean_b:.3f} & "
                f"{r.relative_improvement:+.1f}\\% & {pval} & {sig} \\\\"
            )
        table.append("\\midrule")

    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append(f"\\caption*{{Comparison of {method_a} (A) vs {method_b} (B). "
                 f"$\\Delta$\\% shows relative improvement. Sig. at $\\alpha=0.05$.}}")
    table.append("\\end{table}")

    return "\n".join(table)


def run_all_pairwise_comparisons(
    all_results: Dict[str, Dict[str, Dict[str, List[float]]]],
    method_groups: Optional[Dict[str, List[str]]] = None,
    metrics: Optional[List[str]] = None,
    exclude_outliers: bool = True
) -> Dict[str, List[TestResult]]:
    """
    Run comparisons for predefined method groups.

    Default groups cover:
    - Model selection (baseline vs single-pass models)
    - Pass progression (single → two → three → ... → seven)
    - Pass ordering (diarization-first vs correction-first)
    - Prompting strategy (zero-shot vs few-shot)
    """
    if method_groups is None:
        # Define meaningful comparison groups
        method_groups = {
            # =====================================================================
            # Model Selection: Single-pass comparison (baseline vs each model)
            # =====================================================================
            "baseline vs gpt4omini": ("baseline", "gpt4omini"),
            "baseline vs qwen_vl_8b": ("baseline", "qwen_vl_8b"),
            "baseline vs qwen_80b": ("baseline", "qwen_80b"),

            # =====================================================================
            # Prompting Strategy: Zero-shot vs Few-shot (Qwen80B two-pass)
            # =====================================================================
            "qwen_80b: twopass_zeroshot vs twopass_fewshot": ("qwen_80b_twopass", "qwen_80b_twopass_fewshot"),

            # =====================================================================
            # Pass Ordering: Diarization-first vs Correction-first
            # =====================================================================
            "qwen_80b: twopass vs twopass_reversed": ("qwen_80b_twopass", "qwen_80b_twopass_reversed"),
            "qwen_80b: threepass vs threepass_reversed": ("qwen_80b_threepass", "qwen_80b_threepass_reversed"),

            # =====================================================================
            # Iteration Depth: Pass progression (1 to 7 passes)
            # =====================================================================
            "qwen_80b: single vs twopass": ("qwen_80b", "qwen_80b_twopass"),
            "qwen_80b: twopass vs threepass": ("qwen_80b_twopass", "qwen_80b_threepass"),
            "qwen_80b: threepass vs fourpass": ("qwen_80b_threepass", "qwen_80b_fourpass"),
            "qwen_80b: fourpass vs fivepass": ("qwen_80b_fourpass", "qwen_80b_fivepass"),
            "qwen_80b: fivepass vs sixpass": ("qwen_80b_fivepass", "qwen_80b_sixpass"),
            "qwen_80b: sixpass vs sevenpass": ("qwen_80b_sixpass", "qwen_80b_sevenpass"),

            # =====================================================================
            # Three-pass Diarization-first (3P-D) vs all alternatives
            # =====================================================================
            "3P-D vs baseline": ("qwen_80b_threepass", "baseline"),
            "3P-D vs gpt4omini (1P)": ("qwen_80b_threepass", "gpt4omini"),
            "3P-D vs qwen_vl_8b (1P)": ("qwen_80b_threepass", "qwen_vl_8b"),
            "3P-D vs qwen_80b (1P)": ("qwen_80b_threepass", "qwen_80b"),
            "3P-D vs 2P-D": ("qwen_80b_threepass", "qwen_80b_twopass"),
            "3P-D vs 2P-D-FS": ("qwen_80b_threepass", "qwen_80b_twopass_fewshot"),
            "3P-D vs 3P-C": ("qwen_80b_threepass", "qwen_80b_threepass_reversed"),
            "3P-D vs 4P-D": ("qwen_80b_threepass", "qwen_80b_fourpass"),
            "3P-D vs 5P-D": ("qwen_80b_threepass", "qwen_80b_fivepass"),
            "3P-D vs 6P-D": ("qwen_80b_threepass", "qwen_80b_sixpass"),
            "3P-D vs 7P-D": ("qwen_80b_threepass", "qwen_80b_sevenpass"),

            # =====================================================================
            # Baseline vs all Qwen80B multi-pass configurations
            # =====================================================================
            "baseline vs qwen_80b_twopass": ("baseline", "qwen_80b_twopass"),
            "baseline vs qwen_80b_twopass_fewshot": ("baseline", "qwen_80b_twopass_fewshot"),
            "baseline vs qwen_80b_twopass_reversed": ("baseline", "qwen_80b_twopass_reversed"),
            "baseline vs qwen_80b_threepass": ("baseline", "qwen_80b_threepass"),
            "baseline vs qwen_80b_threepass_reversed": ("baseline", "qwen_80b_threepass_reversed"),
            "baseline vs qwen_80b_fourpass": ("baseline", "qwen_80b_fourpass"),
            "baseline vs qwen_80b_fivepass": ("baseline", "qwen_80b_fivepass"),
            "baseline vs qwen_80b_sixpass": ("baseline", "qwen_80b_sixpass"),
            "baseline vs qwen_80b_sevenpass": ("baseline", "qwen_80b_sevenpass"),
        }

    all_test_results = {}

    for comparison_name, (method_a, method_b) in method_groups.items():
        if method_a not in all_results or method_b not in all_results:
            continue

        print(f"\n>>> {comparison_name}")
        results = compare_methods(
            all_results, method_a, method_b, metrics=metrics,
            exclude_outliers=exclude_outliers
        )
        if results:
            all_test_results[comparison_name] = results

    return all_test_results


def save_results(
    all_test_results: Dict[str, List[TestResult]],
    output_path: Path
):
    """Save all test results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format (handle numpy types)
    data = {}
    for comparison_name, results in all_test_results.items():
        data[comparison_name] = [
            {
                "method_a": r.method_a,
                "method_b": r.method_b,
                "metric": r.metric,
                "domain": r.domain,
                "mean_a": float(r.mean_a),
                "mean_b": float(r.mean_b),
                "std_a": float(r.std_a),
                "std_b": float(r.std_b),
                "difference": float(r.difference),
                "relative_improvement_pct": float(r.relative_improvement),
                "n_samples": int(r.n_samples),
                "wilcoxon_statistic": float(r.wilcoxon_statistic) if r.wilcoxon_statistic is not None else None,
                "wilcoxon_pvalue": float(r.wilcoxon_pvalue) if r.wilcoxon_pvalue is not None else None,
                "significant_wilcoxon_005": bool(r.significant_wilcoxon) if r.significant_wilcoxon is not None else None,
            }
            for r in results
        ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical tests for transcription post-processing methods"
    )
    parser.add_argument(
        "--methods", nargs=2, metavar=("A", "B"),
        help="Compare two specific methods"
    )
    parser.add_argument(
        "--metric", type=str,
        help="Focus on a specific metric (wer, der, wder)"
    )
    parser.add_argument(
        "--domain", type=str,
        help="Focus on a specific domain (neurochirurgie, prevention_suicide)"
    )
    parser.add_argument(
        "--table", action="store_true",
        help="Generate formatted table output"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all predefined pairwise comparisons"
    )
    parser.add_argument(
        "--fourpass-analysis", action="store_true",
        help="Run focused analysis: qwen_80b_fourpass vs other methods (WDER metric only)"
    )
    parser.add_argument(
        "--twopass-analysis", action="store_true",
        help="Run focused analysis: qwen_80b_twopass vs other methods (WDER metric only)"
    )
    parser.add_argument(
        "--threepass-analysis", action="store_true",
        help="Run focused analysis: qwen_80b_threepass vs other methods (WDER metric only)"
    )
    parser.add_argument(
        "--sixpass-analysis", action="store_true",
        help="Run focused analysis: qwen_80b_sixpass vs other methods (WDER metric only)"
    )
    parser.add_argument(
        "--sevenpass-analysis", action="store_true",
        help="Run focused analysis: qwen_80b_sevenpass vs other methods (WDER metric only)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available methods"
    )
    parser.add_argument(
        "--include-outliers", action="store_true",
        help="Include outlier files (metrics > 100%% or < 0%%) in comparisons. Default: exclude outliers."
    )

    args = parser.parse_args()

    # Determine outlier filtering
    exclude_outliers = not args.include_outliers

    if args.list:
        methods = get_available_methods()
        print("\nAvailable methods:")
        for m in methods:
            detailed = (DETAILED_RESULTS_DIR / f"eval_detailed_{m}.json").exists()
            status = "✓ detailed" if detailed else "○ summary only"
            print(f"  - {m} ({status})")
        print("\nNote: Statistical tests require detailed (per-file) results.")
        print("Run: python src/evaluation/evaluate_results.py <method> --save-detailed")
        return

    # Load all detailed results
    all_results = collect_all_detailed_results()

    if not all_results:
        print("\nNo detailed results found!")
        print("\nTo generate detailed results, run evaluate_results.py with --save-detailed:")
        print("  python src/evaluation/evaluate_results.py baseline --save-detailed")
        print("  python src/evaluation/evaluate_results.py gpt4omini_twopass --save-detailed")
        print("  etc.")
        return

    print(f"\nLoaded detailed results for {len(all_results)} methods:")
    for method in sorted(all_results.keys()):
        print(f"  - {method}")

    # Prepare metrics and domains filters
    metrics = [args.metric] if args.metric else None
    domains = [args.domain] if args.domain else None

    # Print outlier filtering status
    if exclude_outliers:
        print("\n[INFO] Outliers (metrics > 100% or < 0%) will be EXCLUDED from comparisons.")
    else:
        print("\n[INFO] Outliers will be INCLUDED in comparisons (--include-outliers).")

    if args.methods:
        # Compare two specific methods
        method_a, method_b = args.methods
        results = compare_methods(
            all_results, method_a, method_b,
            metrics=metrics, domains=domains,
            exclude_outliers=exclude_outliers
        )

        if results:
            single_comparison = {f"{method_a}_vs_{method_b}": results}

            print_results(results)

            if args.table:
                table = generate_table(
                    results,
                    caption=f"Statistical comparison: {method_a} vs {method_b}"
                )
                print("\n" + "=" * 40 + " table " + "=" * 40)
                print(table)

            # Save results
            output_path = OUTPUT_DIR / f"comparison_{method_a}_vs_{method_b}.json"
            save_results(single_comparison, output_path)

    elif args.all:
        # Run all predefined comparisons
        all_test_results = run_all_pairwise_comparisons(
            all_results, metrics=metrics,
            exclude_outliers=exclude_outliers
        )


        for comparison_name, results in all_test_results.items():
            print(f"\n\n{'#' * 80}")
            print(f"# {comparison_name}")
            print(f"{'#' * 80}")
            print_results(results, verbose=False)

        # Save all results
        output_path = OUTPUT_DIR / "all_comparisons.json"
        save_results(all_test_results, output_path)

        # Summary table (uncorrected)
        print("\n\n" + "=" * 95)
        print("SUMMARY: Significant improvements (Wilcoxon p < 0.05)")
        print("=" * 95)
        print(f"{'Comparison':<45} {'Metric':<8} {'Domain':<20} {'Δ%':>8} {'p-value':>12}")
        print("-" * 95)

        for comparison_name, results in all_test_results.items():
            for r in results:
                if r.significant_wilcoxon and r.relative_improvement > 0:
                    pval_str = f"{r.wilcoxon_pvalue:.6f}" if r.wilcoxon_pvalue else "N/A"
                    print(f"{comparison_name:<45} {r.metric:<8} {r.domain:<20} "
                          f"{r.relative_improvement:>+7.1f}% {pval_str:>12}")

        

    elif args.fourpass_analysis:
        # Focused analysis: qwen_80b_fourpass vs 11 specific methods
        print("\n" + "=" * 80)
        print("FOURPASS ANALYSIS: qwen_80b_fourpass vs 11 methods (WDER metric only)")
        print("=" * 80)

        fourpass_comparisons = {
            # 1. Baseline
            "fourpass vs baseline": ("qwen_80b_fourpass", "baseline"),
            # 2-4. Single-pass models
            "fourpass vs gpt4omini (single)": ("qwen_80b_fourpass", "gpt4omini"),
            "fourpass vs qwen_vl_8b (single)": ("qwen_80b_fourpass", "qwen_vl_8b"),
            "fourpass vs qwen_80b (single)": ("qwen_80b_fourpass", "qwen_80b"),
            # 5. Fewshot
            "fourpass vs qwen_80b_twopass_fewshot": ("qwen_80b_fourpass", "qwen_80b_twopass_fewshot"),
            # 6-10. Pass progression
            "fourpass vs qwen_80b_twopass": ("qwen_80b_fourpass", "qwen_80b_twopass"),
            "fourpass vs qwen_80b_threepass": ("qwen_80b_fourpass", "qwen_80b_threepass"),
            "fourpass vs qwen_80b_fivepass": ("qwen_80b_fourpass", "qwen_80b_fivepass"),
            "fourpass vs qwen_80b_sixpass": ("qwen_80b_fourpass", "qwen_80b_sixpass"),
            "fourpass vs qwen_80b_sevenpass": ("qwen_80b_fourpass", "qwen_80b_sevenpass"),
        }

        # Force WDER metric only for fourpass analysis
        fourpass_metrics = ["wder"]
        print(f"\nRunning {len(fourpass_comparisons)} comparisons × 1 metric (WDER) × 2 domains = {len(fourpass_comparisons) * 2} tests")

        all_test_results = {}
        for comparison_name, (method_a, method_b) in fourpass_comparisons.items():
            if method_a not in all_results or method_b not in all_results:
                print(f"  [SKIP] {comparison_name} - missing data for {method_a if method_a not in all_results else method_b}")
                continue

            print(f"\n>>> {comparison_name}")
            results = compare_methods(
                all_results, method_a, method_b, metrics=fourpass_metrics,
                exclude_outliers=exclude_outliers
            )
            if results:
                all_test_results[comparison_name] = results

        # Print detailed results
        for comparison_name, results in all_test_results.items():
            print(f"\n\n{'#' * 80}")
            print(f"# {comparison_name}")
            print(f"{'#' * 80}")
            print_results(results, verbose=False)

        # Save results
        output_path = OUTPUT_DIR / "fourpass_analysis.json"
        save_results(all_test_results, output_path)

        # Summary table (uncorrected)
        print("\n\n" + "=" * 100)
        print("FOURPASS ANALYSIS - WDER (Wilcoxon p < 0.05, UNCORRECTED)")
        print("=" * 100)
        print(f"{'Comparison':<40} {'Domain':<20} {'fourpass WDER':>14} {'other WDER':>12} {'Δ%':>8} {'p-value':>12}")
        print("-" * 100)

        for comparison_name, results in all_test_results.items():
            for r in results:
                if r.significant_wilcoxon:
                    pval_str = f"{r.wilcoxon_pvalue:.6f}" if r.wilcoxon_pvalue else "N/A"
                    print(f"{comparison_name:<40} {r.domain:<20} "
                          f"{r.mean_a:>14.4f} {r.mean_b:>12.4f} {r.relative_improvement:>+7.1f}% {pval_str:>12}")

        

        # Count comparisons and p-values
        total_tests = sum(len(results) for results in all_test_results.values())
        print(f"\n\nTotal statistical tests: {total_tests}")

    elif args.twopass_analysis:
        # Focused analysis: qwen_80b_twopass vs other methods
        print("\n" + "=" * 80)
        print("TWOPASS ANALYSIS: qwen_80b_twopass vs other methods (WDER metric only)")
        print("=" * 80)

        twopass_comparisons = {
            # 1. Baseline
            "twopass vs baseline": ("qwen_80b_twopass", "baseline"),
            # 2-4. Single-pass models
            "twopass vs gpt4omini (single)": ("qwen_80b_twopass", "gpt4omini"),
            "twopass vs qwen_vl_8b (single)": ("qwen_80b_twopass", "qwen_vl_8b"),
            "twopass vs qwen_80b (single)": ("qwen_80b_twopass", "qwen_80b"),
            # 5. Fewshot
            "twopass vs qwen_80b_twopass_fewshot": ("qwen_80b_twopass", "qwen_80b_twopass_fewshot"),
            # 6-10. Pass progression (excluding twopass itself)
            "twopass vs qwen_80b_threepass": ("qwen_80b_twopass", "qwen_80b_threepass"),
            "twopass vs qwen_80b_fourpass": ("qwen_80b_twopass", "qwen_80b_fourpass"),
            "twopass vs qwen_80b_fivepass": ("qwen_80b_twopass", "qwen_80b_fivepass"),
            "twopass vs qwen_80b_sixpass": ("qwen_80b_twopass", "qwen_80b_sixpass"),
            "twopass vs qwen_80b_sevenpass": ("qwen_80b_twopass", "qwen_80b_sevenpass"),
        }

        # Force WDER metric only
        twopass_metrics = ["wder"]
        print(f"\nRunning {len(twopass_comparisons)} comparisons × 1 metric (WDER) × 2 domains = {len(twopass_comparisons) * 2} tests")

        all_test_results = {}
        for comparison_name, (method_a, method_b) in twopass_comparisons.items():
            if method_a not in all_results or method_b not in all_results:
                print(f"  [SKIP] {comparison_name} - missing data for {method_a if method_a not in all_results else method_b}")
                continue

            print(f"\n>>> {comparison_name}")
            results = compare_methods(
                all_results, method_a, method_b, metrics=twopass_metrics,
                exclude_outliers=exclude_outliers
            )
            if results:
                all_test_results[comparison_name] = results

        # Print detailed results
        for comparison_name, results in all_test_results.items():
            print(f"\n\n{'#' * 80}")
            print(f"# {comparison_name}")
            print(f"{'#' * 80}")
            print_results(results, verbose=False)

        # Save results
        output_path = OUTPUT_DIR / "twopass_analysis.json"
        save_results(all_test_results, output_path)

        # Summary table (uncorrected)
        print("\n\n" + "=" * 100)
        print("TWOPASS ANALYSIS - WDER (Wilcoxon p < 0.05, UNCORRECTED)")
        print("=" * 100)
        print(f"{'Comparison':<40} {'Domain':<20} {'twopass WDER':>14} {'other WDER':>12} {'Δ%':>8} {'p-value':>12}")
        print("-" * 100)

        for comparison_name, results in all_test_results.items():
            for r in results:
                if r.significant_wilcoxon:
                    pval_str = f"{r.wilcoxon_pvalue:.6f}" if r.wilcoxon_pvalue else "N/A"
                    print(f"{comparison_name:<40} {r.domain:<20} "
                          f"{r.mean_a:>14.4f} {r.mean_b:>12.4f} {r.relative_improvement:>+7.1f}% {pval_str:>12}")

        total_tests = sum(len(results) for results in all_test_results.values())
        print(f"\n\nTotal statistical tests: {total_tests}")

    elif args.threepass_analysis:
        # Focused analysis: qwen_80b_threepass vs other methods
        print("\n" + "=" * 80)
        print("THREEPASS ANALYSIS: qwen_80b_threepass vs other methods (WDER metric only)")
        print("=" * 80)

        threepass_comparisons = {
            # 1. Baseline
            "threepass vs baseline": ("qwen_80b_threepass", "baseline"),
            # 2-4. Single-pass models (all LLMs)
            "threepass vs gpt4omini (single)": ("qwen_80b_threepass", "gpt4omini"),
            "threepass vs qwen_vl_8b (single)": ("qwen_80b_threepass", "qwen_vl_8b"),
            "threepass vs qwen_80b (single)": ("qwen_80b_threepass", "qwen_80b"),
            # 5. Fewshot two-pass
            "threepass vs qwen_80b_twopass_fewshot": ("qwen_80b_threepass", "qwen_80b_twopass_fewshot"),
            # 6. Reversed order (correction-first vs diarization-first)
            "threepass vs threepass_reversed": ("qwen_80b_threepass", "qwen_80b_threepass_reversed"),
            # 7-12. All pass configurations of Qwen 80B (excluding threepass itself)
            "threepass vs qwen_80b_twopass": ("qwen_80b_threepass", "qwen_80b_twopass"),
            "threepass vs qwen_80b_fourpass": ("qwen_80b_threepass", "qwen_80b_fourpass"),
            "threepass vs qwen_80b_fivepass": ("qwen_80b_threepass", "qwen_80b_fivepass"),
            "threepass vs qwen_80b_sixpass": ("qwen_80b_threepass", "qwen_80b_sixpass"),
            "threepass vs qwen_80b_sevenpass": ("qwen_80b_threepass", "qwen_80b_sevenpass"),
        }

        # Force WDER metric only
        threepass_metrics = ["wder"]
        print(f"\nRunning {len(threepass_comparisons)} comparisons × 1 metric (WDER) × 2 domains = {len(threepass_comparisons) * 2} tests")

        all_test_results = {}
        for comparison_name, (method_a, method_b) in threepass_comparisons.items():
            if method_a not in all_results or method_b not in all_results:
                print(f"  [SKIP] {comparison_name} - missing data for {method_a if method_a not in all_results else method_b}")
                continue

            print(f"\n>>> {comparison_name}")
            results = compare_methods(
                all_results, method_a, method_b, metrics=threepass_metrics,
                exclude_outliers=exclude_outliers
            )
            if results:
                all_test_results[comparison_name] = results

        # Print detailed results
        for comparison_name, results in all_test_results.items():
            print(f"\n\n{'#' * 80}")
            print(f"# {comparison_name}")
            print(f"{'#' * 80}")
            print_results(results, verbose=False)

        # Save results
        output_path = OUTPUT_DIR / "threepass_analysis.json"
        save_results(all_test_results, output_path)

        # Summary table (significant only)
        print("\n\n" + "=" * 100)
        print("THREEPASS ANALYSIS - WDER (Wilcoxon p < 0.05, UNCORRECTED)")
        print("=" * 100)
        print(f"{'Comparison':<40} {'Domain':<20} {'threepass WDER':>14} {'other WDER':>12} {'Δ%':>8} {'p-value':>12}")
        print("-" * 100)

        for comparison_name, results in all_test_results.items():
            for r in results:
                if r.significant_wilcoxon:
                    pval_str = f"{r.wilcoxon_pvalue:.6f}" if r.wilcoxon_pvalue else "N/A"
                    print(f"{comparison_name:<40} {r.domain:<20} "
                          f"{r.mean_a:>14.4f} {r.mean_b:>12.4f} {r.relative_improvement:>+7.1f}% {pval_str:>12}")

        total_tests = sum(len(results) for results in all_test_results.values())
        print(f"\n\nTotal statistical tests: {total_tests}")

    elif args.sixpass_analysis:
        # Focused analysis: qwen_80b_sixpass vs other methods
        print("\n" + "=" * 80)
        print("SIXPASS ANALYSIS: qwen_80b_sixpass vs other methods (WDER metric only)")
        print("=" * 80)

        sixpass_comparisons = {
            # 1. Baseline
            "sixpass vs baseline": ("qwen_80b_sixpass", "baseline"),
            # 2-4. Single-pass models
            "sixpass vs gpt4omini (single)": ("qwen_80b_sixpass", "gpt4omini"),
            "sixpass vs qwen_vl_8b (single)": ("qwen_80b_sixpass", "qwen_vl_8b"),
            "sixpass vs qwen_80b (single)": ("qwen_80b_sixpass", "qwen_80b"),
            # 5. Fewshot
            "sixpass vs qwen_80b_twopass_fewshot": ("qwen_80b_sixpass", "qwen_80b_twopass_fewshot"),
            # 6-10. Pass progression (excluding sixpass itself)
            "sixpass vs qwen_80b_twopass": ("qwen_80b_sixpass", "qwen_80b_twopass"),
            "sixpass vs qwen_80b_threepass": ("qwen_80b_sixpass", "qwen_80b_threepass"),
            "sixpass vs qwen_80b_fourpass": ("qwen_80b_sixpass", "qwen_80b_fourpass"),
            "sixpass vs qwen_80b_fivepass": ("qwen_80b_sixpass", "qwen_80b_fivepass"),
            "sixpass vs qwen_80b_sevenpass": ("qwen_80b_sixpass", "qwen_80b_sevenpass"),
        }

        # Force WDER metric only
        sixpass_metrics = ["wder"]
        print(f"\nRunning {len(sixpass_comparisons)} comparisons × 1 metric (WDER) × 2 domains = {len(sixpass_comparisons) * 2} tests")

        all_test_results = {}
        for comparison_name, (method_a, method_b) in sixpass_comparisons.items():
            if method_a not in all_results or method_b not in all_results:
                print(f"  [SKIP] {comparison_name} - missing data for {method_a if method_a not in all_results else method_b}")
                continue

            print(f"\n>>> {comparison_name}")
            results = compare_methods(
                all_results, method_a, method_b, metrics=sixpass_metrics,
                exclude_outliers=exclude_outliers
            )
            if results:
                all_test_results[comparison_name] = results

        # Print detailed results
        for comparison_name, results in all_test_results.items():
            print(f"\n\n{'#' * 80}")
            print(f"# {comparison_name}")
            print(f"{'#' * 80}")
            print_results(results, verbose=False)

        # Save results
        output_path = OUTPUT_DIR / "sixpass_analysis.json"
        save_results(all_test_results, output_path)

        # Summary table (uncorrected)
        print("\n\n" + "=" * 100)
        print("SIXPASS ANALYSIS - WDER (Wilcoxon p < 0.05, UNCORRECTED)")
        print("=" * 100)
        print(f"{'Comparison':<40} {'Domain':<20} {'sixpass WDER':>14} {'other WDER':>12} {'Δ%':>8} {'p-value':>12}")
        print("-" * 100)

        for comparison_name, results in all_test_results.items():
            for r in results:
                if r.significant_wilcoxon:
                    pval_str = f"{r.wilcoxon_pvalue:.6f}" if r.wilcoxon_pvalue else "N/A"
                    print(f"{comparison_name:<40} {r.domain:<20} "
                          f"{r.mean_a:>14.4f} {r.mean_b:>12.4f} {r.relative_improvement:>+7.1f}% {pval_str:>12}")

        total_tests = sum(len(results) for results in all_test_results.values())
        print(f"\n\nTotal statistical tests: {total_tests}")

    elif args.sevenpass_analysis:
        # Focused analysis: qwen_80b_sevenpass vs other methods
        print("\n" + "=" * 80)
        print("SEVENPASS ANALYSIS: qwen_80b_sevenpass vs other methods (WDER metric only)")
        print("=" * 80)

        sevenpass_comparisons = {
            # 1. Baseline
            "sevenpass vs baseline": ("qwen_80b_sevenpass", "baseline"),
            # 2-4. Single-pass models
            "sevenpass vs gpt4omini (single)": ("qwen_80b_sevenpass", "gpt4omini"),
            "sevenpass vs qwen_vl_8b (single)": ("qwen_80b_sevenpass", "qwen_vl_8b"),
            "sevenpass vs qwen_80b (single)": ("qwen_80b_sevenpass", "qwen_80b"),
            # 5. Fewshot
            "sevenpass vs qwen_80b_twopass_fewshot": ("qwen_80b_sevenpass", "qwen_80b_twopass_fewshot"),
            # 6-10. Pass progression (excluding sevenpass itself)
            "sevenpass vs qwen_80b_twopass": ("qwen_80b_sevenpass", "qwen_80b_twopass"),
            "sevenpass vs qwen_80b_threepass": ("qwen_80b_sevenpass", "qwen_80b_threepass"),
            "sevenpass vs qwen_80b_fourpass": ("qwen_80b_sevenpass", "qwen_80b_fourpass"),
            "sevenpass vs qwen_80b_fivepass": ("qwen_80b_sevenpass", "qwen_80b_fivepass"),
            "sevenpass vs qwen_80b_sixpass": ("qwen_80b_sevenpass", "qwen_80b_sixpass"),
        }

        # Force WDER metric only
        sevenpass_metrics = ["wder"]
        print(f"\nRunning {len(sevenpass_comparisons)} comparisons × 1 metric (WDER) × 2 domains = {len(sevenpass_comparisons) * 2} tests")

        all_test_results = {}
        for comparison_name, (method_a, method_b) in sevenpass_comparisons.items():
            if method_a not in all_results or method_b not in all_results:
                print(f"  [SKIP] {comparison_name} - missing data for {method_a if method_a not in all_results else method_b}")
                continue

            print(f"\n>>> {comparison_name}")
            results = compare_methods(
                all_results, method_a, method_b, metrics=sevenpass_metrics,
                exclude_outliers=exclude_outliers
            )
            if results:
                all_test_results[comparison_name] = results

        # Print detailed results
        for comparison_name, results in all_test_results.items():
            print(f"\n\n{'#' * 80}")
            print(f"# {comparison_name}")
            print(f"{'#' * 80}")
            print_results(results, verbose=False)

        # Save results
        output_path = OUTPUT_DIR / "sevenpass_analysis.json"
        save_results(all_test_results, output_path)

        # Summary table (uncorrected)
        print("\n\n" + "=" * 100)
        print("SEVENPASS ANALYSIS - WDER (Wilcoxon p < 0.05, UNCORRECTED)")
        print("=" * 100)
        print(f"{'Comparison':<40} {'Domain':<20} {'sevenpass WDER':>14} {'other WDER':>12} {'Δ%':>8} {'p-value':>12}")
        print("-" * 100)

        for comparison_name, results in all_test_results.items():
            for r in results:
                if r.significant_wilcoxon:
                    pval_str = f"{r.wilcoxon_pvalue:.6f}" if r.wilcoxon_pvalue else "N/A"
                    print(f"{comparison_name:<40} {r.domain:<20} "
                          f"{r.mean_a:>14.4f} {r.mean_b:>12.4f} {r.relative_improvement:>+7.1f}% {pval_str:>12}")

        total_tests = sum(len(results) for results in all_test_results.values())
        print(f"\n\nTotal statistical tests: {total_tests}")

    else:
        # Default: show usage
        parser.print_help()
        print("\n\nExample usage:")
        print("  python statistical_tests.py --list")
        print("  python statistical_tests.py --methods baseline gpt4omini_twopass")
        print("  python statistical_tests.py --all")
        print("  python statistical_tests.py --twopass-analysis")
        print("  python statistical_tests.py --threepass-analysis")
        print("  python statistical_tests.py --fourpass-analysis")
        print("  python statistical_tests.py --sixpass-analysis")
        print("  python statistical_tests.py --sevenpass-analysis")


if __name__ == "__main__":
    main()