from __future__ import annotations
import argparse
from pathlib import Path
from .benchmark_runner import run_benchmark, summarize, save
from .experiment_lab import WORMBench


def main() -> int:
    ap = argparse.ArgumentParser(description='WORM FARM v2 synthetic benchmark runner')
    ap.add_argument('--tasks-per-family', type=int, default=1)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--generations', type=int, default=5)
    ap.add_argument('--output', default='worm_bench_results')
    args = ap.parse_args()
    bench = WORMBench()
    tasks = bench.make_tasks(max(1, args.tasks_per_family))
    seeds = [101 + i * 2 for i in range(max(1, args.seeds))]
    architectures = ["single","multi","adversarial","worm_v2","worm_v2_no_genetics","worm_v2_no_arena","worm_v2_no_harsh"]
    results = run_benchmark(tasks=tasks, seeds=seeds, generations=max(1, args.generations), architectures=architectures)
    summaries = summarize(results)
    save(results, summaries, Path(args.output))
    for s in summaries:
        print(s)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
