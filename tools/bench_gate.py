#!/usr/bin/env python3
"""Bench gate: an A/B of the base revision and the PR on the same runner (M5, DECISIONS.md #24).

Adapted from pragma.component-ecs's tools/bench_gate.py (its DECISIONS #32, #43, #44), itself
adapted from pragma.core's (its DECISIONS #34).

Both revisions are built up front, then measurements are interleaved: base → head → base → head …
(`--rounds` rounds). For each measurement the best (minimum) criterion median across rounds
is taken: shared-runner noise only slows things down, so the minimum is the most
stable estimate.

A regression is computed as follows:
  ratio      = head / base for the measurement itself;
  ref_ratio  = head / base for its reference (`<group>/hashmap_churn/1000`, a std `HashMap` in
               the same bench): the change does not affect it, so its shift is runner noise;
  normalized = ratio / ref_ratio.
A measurement fails if ratio > limit AND normalized > limit, where limit = 1 + threshold + spread
and spread is the measurement's own process-to-process spread (max / min − 1 over the rounds of
each revision, the larger of the two).

A measurement missing on the base revision (a new benchmark) is reported, not gated; so is every
measurement while the base revision has no benchmarks at all (the PR that adds them).

Both revisions are built with `-C llvm-args=-align-loops=64` added to RUSTFLAGS, so that code
placement does not decide a measurement. Every measurement runs in its own process, and on Linux
with MALLOC_ARENA_MAX=1, so that the allocator state does not either.

Usage:
  python3 tools/bench_gate.py --base <base worktree> --head <head worktree>
                              [--rounds 3] [--threshold 0.10] [--summary FILE]
Exit code 1 on a regression, 2 on a launch error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# (package, bench target, criterion filter).
BENCHES = [
    ("pragma_rhi", "resources", r"^resources/"),
    ("pragma_rhi", "recording", r"^recording/"),
    ("pragma_rhi", "submission", r"^submission/"),
]
CRITERION_ARGS = ["--warm-up-time", "1", "--measurement-time", "3", "--noplot"]

# Measurement → reference that pragma_rhi changes do not affect.
REFERENCES = {
    "resources/buffer_churn/1000": "resources/hashmap_churn/1000",
    "resources/texture_view_churn/1000": "resources/hashmap_churn/1000",
    "resources/bind_group_churn/1000": "resources/hashmap_churn/1000",
    "recording/encode_draws/1000": "recording/hashmap_churn/1000",
    "recording/encode_copies/1000": "recording/hashmap_churn/1000",
    "recording/graph_8_tasks/1000": "recording/hashmap_churn/1000",
    "submission/submit_draws/1000": "submission/hashmap_churn/1000",
    "submission/submit_copies/1000": "submission/hashmap_churn/1000",
    "submission/frame_draws/1000": "submission/hashmap_churn/1000",
}
# Every measurement the gate uses (gated ones and their references) with its bench target; each
# runs in a process of its own.
MEASUREMENTS = [
    (package, bench, bench_id)
    for bench_id in sorted(set(REFERENCES) | set(REFERENCES.values()))
    for package, bench, pattern in BENCHES
    if re.match(pattern, bench_id)
]

# Both sides are built with hot loops aligned to 64 bytes: otherwise where a loop lands in the
# binary can decide a measurement on a shared runner, and any unrelated code change moves it.
ALIGN_LOOPS = "-C llvm-args=-align-loops=64"


@dataclass
class Side:
    name: str
    root: Path
    target: Path

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CARGO_TARGET_DIR"] = str(self.target)
        env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} {ALIGN_LOOPS}".strip()
        if sys.platform.startswith("linux"):
            env["MALLOC_ARENA_MAX"] = "1"  # one malloc arena: pool threads share the allocator state
        return env

    def has_benches(self) -> bool:
        """Whether this revision has the benchmarks (the base of the PR adding them has not)."""
        return all(
            (self.root / "crates" / package / "benches" / f"{bench}.rs").exists()
            for package, bench, _ in BENCHES
        )


def run(cmd: list[str], side: Side) -> str:
    print(f"[{side.name}] $ {' '.join(cmd)}", flush=True)
    done = subprocess.run(
        cmd, cwd=side.root, env=side.env(), capture_output=True, text=True, encoding="utf-8"
    )
    if done.returncode != 0:
        sys.stderr.write(done.stdout + done.stderr)
        raise SystemExit(2)
    return done.stdout


def criterion_results(side: Side) -> dict[str, float]:
    """Medians (ns) of the last run: target/criterion/<id>/new/estimates.json."""
    base = side.target / "criterion"
    results = {}
    for path in base.glob("**/new/estimates.json"):
        bench_id = path.parent.parent.relative_to(base).as_posix()
        data = json.loads(path.read_text(encoding="utf-8"))
        results[bench_id] = float(data["median"]["point_estimate"])
    return results


def build(side: Side) -> None:
    for package, bench, _ in BENCHES:
        run(["cargo", "bench", "-p", package, "--bench", bench, "--no-run"], side)


def measure(side: Side) -> dict[str, float]:
    """The measurements of one round."""
    # One process per measurement: the allocator state left by the benchmarks run before it in
    # the same process must not decide a measurement.
    for package, bench, bench_id in MEASUREMENTS:
        run(
            [
                "cargo", "bench", "-p", package, "--bench", bench, "--",
                f"^{re.escape(bench_id)}$", *CRITERION_ARGS,
            ],
            side,
        )
    return criterion_results(side)


def process_spread(rounds: list[dict[str, float]], key: str) -> float:
    """How far one revision's runs of `key` (one process each) lie apart: max / min − 1 (DECISIONS #44)."""
    values = [r[key] for r in rounds if key in r]
    return max(values) / min(values) - 1.0 if values else 0.0


def best(rounds: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for result in rounds:
        for key, value in result.items():
            merged[key] = min(value, merged.get(key, value))
    return merged


def fmt_ns(ns: float) -> str:
    if ns >= 1e6:
        return f"{ns / 1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.2f} µs"
    return f"{ns:.1f} ns"


def main() -> int:
    # The Windows runner console is cp1252: non-Latin-1 report characters (≤, →) would crash the script.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    head_root = args.head.resolve()
    sides = [
        Side("base", args.base.resolve(), head_root / "target" / "bench-gate-base"),
        Side("head", head_root, head_root / "target" / "bench-gate-head"),
    ]
    # The base of the PR that adds the benchmarks has none: everything is reported, not gated.
    sides = [side for side in sides if side.has_benches()]
    for side in sides:
        build(side)
    rounds: dict[str, list[dict[str, float]]] = {"base": [], "head": []}
    for r in range(args.rounds):
        for side in sides:
            print(f"--- round {r + 1}/{args.rounds}: {side.name}", flush=True)
            rounds[side.name].append(measure(side))
    base, head = best(rounds["base"]), best(rounds["head"])

    ref_ratios = {
        ref: head[ref] / base[ref] for ref in set(REFERENCES.values()) if ref in base and ref in head
    }
    noise = max((abs(r - 1.0) for r in ref_ratios.values()), default=0.0)
    limit = 1.0 + args.threshold

    rows, failures = [], []
    gated = sorted(k for k in head if k in REFERENCES)
    for key in gated:
        if key not in base:
            rows.append(f"| `{key}` | — | {fmt_ns(head[key])} | — | new | — | not gated |")
            continue
        ratio = head[key] / base[key]
        ref = REFERENCES.get(key)
        if ref in ref_ratios:
            normalized = ratio / ref_ratios[ref]
            # A measurement that lands in a different mode per process (DECISIONS #44) cannot be
            # gated tighter than its own process-to-process spread.
            spread_k = max(process_spread(rounds["base"], key), process_spread(rounds["head"], key))
            key_limit = limit + spread_k
            failed = ratio > key_limit and normalized > key_limit
            note = f"{ref_ratios[ref]:.3f} ({ref.split('/', 2)[1]}); limit {key_limit:.3f}"
        else:
            normalized = ratio
            failed = False
            note = "reference missing on one side; not gated"
        verdict = "REGRESSION" if failed else "ok"
        if failed:
            failures.append(key)
        rows.append(
            f"| `{key}` | {fmt_ns(base[key])} | {fmt_ns(head[key])} | {ratio:.3f} | "
            f"{note} | {normalized:.3f} | {verdict} |"
        )

    lines = [
        f"### Bench gate ({os.cpu_count()} cores, {args.rounds} A/B rounds, threshold +{args.threshold:.0%})",
        "",
        "| Measurement | base | head | head/base | reference head/base | normalized | verdict |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"Runner noise (largest reference shift): {noise:.1%}.",
    ]
    report = "\n".join(lines)
    print(report)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as fh:
            fh.write(report + "\n")
    if failures:
        print(f"::error::regression > {args.threshold:.0%}: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
