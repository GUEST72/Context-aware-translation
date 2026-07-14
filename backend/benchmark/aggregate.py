#!/usr/bin/env python3
"""Aggregates raw per-trial JSON (runs/<variant>/<load>/trial_*.json) into a
mean +/- stddev summary table, and writes plots (latency/throughput/health
vs concurrency).

Warm-up runs (warmup_*.json) are intentionally excluded.
"""
from __future__ import annotations

import json
import os
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH_DIR = Path(__file__).resolve().parent
RUNS_DIR = BENCH_DIR / "runs"
VARIANTS = ["a", "b", "c"]
VARIANT_LABELS = {"a": "A: async inline", "b": "B: threadpool", "c": "C: celery"}
VARIANT_COLORS = {"a": "#d64545", "b": "#3b82c4", "c": "#3fa34d"}


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], 0.0)
    return (stats.mean(values), stats.stdev(values))


def load_trials(variant: str, load: int) -> list[dict]:
    level_dir = RUNS_DIR / variant / str(load)
    trials = []
    for f in sorted(level_dir.glob("trial_*.json")):
        trials.append(json.loads(f.read_text()))
    return trials


def build_table() -> dict:
    table = {}
    for variant in VARIANTS:
        variant_dir = RUNS_DIR / variant
        if not variant_dir.exists():
            continue
        loads = sorted(int(p.name) for p in variant_dir.iterdir() if p.is_dir())
        table[variant] = {}
        for load in loads:
            trials = load_trials(variant, load)
            if not trials:
                continue
            summaries = [t["summary"] for t in trials]

            def metric(path_fn):
                return [path_fn(s) for s in summaries]

            wall = metric(lambda s: s["wall_clock_s"])
            throughput = metric(lambda s: s["throughput_rps"])
            p50 = metric(lambda s: s["latency_ms"]["p50"])
            p95 = metric(lambda s: s["latency_ms"]["p95"])
            p99 = metric(lambda s: s["latency_ms"]["p99"])
            lat_max = metric(lambda s: s["latency_ms"]["max"])
            lat_min = metric(lambda s: s["latency_ms"]["min"])
            success = metric(lambda s: s["success_count"])
            timeout = metric(lambda s: s["timeout_count"])
            error = metric(lambda s: s["error_count"])
            health_min = metric(lambda s: s["health_probe"]["latency_ms"]["min"])
            health_p50 = metric(lambda s: s["health_probe"]["latency_ms"]["p50"])
            health_p95 = metric(lambda s: s["health_probe"]["latency_ms"]["p95"])
            health_p99 = metric(lambda s: s["health_probe"]["latency_ms"]["p99"])
            health_max = metric(lambda s: s["health_probe"]["latency_ms"]["max"])
            cpu = metric(lambda s: s["resources"]["peak_cpu_percent_sum"])
            cpu_per_core = metric(lambda s: s["resources"]["peak_cpu_percent_sum"] / (os.cpu_count() or 1))
            rss = metric(lambda s: s["resources"]["peak_rss_mb"])
            threads = metric(lambda s: s["resources"]["peak_thread_count"])
            procs = metric(lambda s: s["resources"]["peak_process_count"])
            total_requests = metric(lambda s: s["success_count"] + s["timeout_count"] + s["error_count"])

            table[variant][load] = {
                "n_trials": len(trials),
                "wall_clock_s": mean_std(wall),
                "throughput_rps": mean_std(throughput),
                "latency_min_ms": mean_std(lat_min),
                "latency_p50_ms": mean_std(p50),
                "latency_p95_ms": mean_std(p95),
                "latency_p99_ms": mean_std(p99),
                "latency_max_ms": mean_std(lat_max),
                "success_count": mean_std(success),
                "timeout_count": mean_std(timeout),
                "error_count": mean_std(error),
                "success_rate_pct": mean_std([100 * s / t if t else 0.0 for s, t in zip(success, total_requests)]),
                "timeout_rate_pct": mean_std([100 * s / t if t else 0.0 for s, t in zip(timeout, total_requests)]),
                "error_rate_pct": mean_std([100 * s / t if t else 0.0 for s, t in zip(error, total_requests)]),
                "health_min_ms": mean_std(health_min),
                "health_p50_ms": mean_std(health_p50),
                "health_p99_ms": mean_std(health_p99),
                "peak_cpu_percent_per_core": mean_std(cpu_per_core),
                "health_p95_ms": mean_std(health_p95),
                "health_max_ms": mean_std(health_max),
                "peak_cpu_percent_sum": mean_std(cpu),
                "peak_rss_mb": mean_std(rss),
                "peak_thread_count": mean_std(threads),
                "peak_process_count": mean_std(procs),
            }
    return table


def fmt(m: tuple[float, float], nd: int = 1) -> str:
    return f"{m[0]:.{nd}f} +/- {m[1]:.{nd}f}"


def print_table(table: dict) -> None:
    header = (f"{'variant':8} {'load':>6} {'wall_s':>16} {'thpt_rps':>14} {'p50_ms':>16} "
              f"{'p95_ms':>16} {'p99_ms':>16} {'max_ms':>16} {'health_p95_ms':>18} "
              f"{'succ/timeout/err':>18} {'peak_cpu%':>14} {'peak_rss_mb':>16} {'peak_thr/proc':>14}")
    print(header)
    print("-" * len(header))
    for variant in VARIANTS:
        if variant not in table:
            continue
        for load, m in table[variant].items():
            succ = f"{m['success_count'][0]:.0f}/{m['timeout_count'][0]:.0f}/{m['error_count'][0]:.0f}"
            tp = f"{m['peak_thread_count'][0]:.0f}/{m['peak_process_count'][0]:.0f}"
            print(f"{variant:8} {load:>6} {fmt(m['wall_clock_s'],2):>16} {fmt(m['throughput_rps'],2):>14} "
                  f"{fmt(m['latency_p50_ms']):>16} {fmt(m['latency_p95_ms']):>16} "
                  f"{fmt(m['latency_p99_ms']):>16} {fmt(m['latency_max_ms']):>16} "
                  f"{fmt(m['health_p95_ms']):>18} {succ:>18} {fmt(m['peak_cpu_percent_sum']):>14} "
                  f"{fmt(m['peak_rss_mb']):>16} {tp:>14}")


def plot_metric_vs_concurrency(table: dict, metric_keys: list[str], labels: list[str], title: str,
                                ylabel: str, out_path: Path, logy: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in VARIANTS:
        if variant not in table:
            continue
        loads = sorted(table[variant].keys())
        for metric_key, label, style in zip(metric_keys, labels, ["-o", "--s", ":^"]):
            means = [table[variant][l][metric_key][0] for l in loads]
            stds = [table[variant][l][metric_key][1] for l in loads]
            series_label = f"{VARIANT_LABELS[variant]} {label}".strip()
            ax.errorbar(loads, means, yerr=stds, fmt=style, color=VARIANT_COLORS[variant],
                        alpha=1.0 if len(metric_keys) == 1 else (1.0 if label == labels[0] else 0.55),
                        label=series_label, capsize=3)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Concurrency (concurrent requests)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    table = build_table()
    print_table(table)

    out = RUNS_DIR / "summary.json"
    out.write_text(json.dumps(table, indent=2, default=str))
    print(f"\nsummary written to {out}")

    plots_dir = RUNS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_metric_vs_concurrency(
        table, ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"], ["p50", "p95", "p99"],
        "Request latency (upload_pdf) vs concurrency", "Latency (ms, log scale)",
        plots_dir / "latency_vs_concurrency.png", logy=True,
    )
    plot_metric_vs_concurrency(
        table, ["throughput_rps"], [""],
        "Throughput vs concurrency", "Requests / second",
        plots_dir / "throughput_vs_concurrency.png",
    )
    plot_metric_vs_concurrency(
        table, ["health_p95_ms"], ["/health p95"],
        "Event-loop responsiveness (/health latency) vs concurrency", "Health-check latency (ms, log scale)",
        plots_dir / "health_latency_vs_concurrency.png", logy=True,
    )
    print(f"plots written to {plots_dir}")


if __name__ == "__main__":
    main()
