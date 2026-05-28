"""BCS core pipeline benchmark with per-stage timings.

Measures the byte-substrate -> field evolution -> clustering -> token discovery
path used by the quick integration tests. The output is intentionally compact so
it can be pasted into regression notes.
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bcs.core.substrate import ByteSubstrate
from bcs.core.field import FieldSystemV6
from bcs.perception.organization import SelfOrganizerV4
from bcs.perception.token import EmergentTokenDiscovery


TEXT_SEED = (
    "BCS cognitive architecture observes byte streams, segments patterns, "
    "keeps UTF-8 boundaries intact, and discovers repeated structures. "
    "Привіт світ. Дані мають повтори, межі, токени і контекст. "
)


def make_data(size: int, mode: str) -> bytes:
    if mode == "text":
        raw = (TEXT_SEED * ((size // len(TEXT_SEED.encode("utf-8"))) + 2)).encode("utf-8")
        return raw[:size]
    if mode == "binary":
        rng = random.Random(42)
        return bytes(rng.randrange(0, 256) for _ in range(size))
    if mode == "structured":
        unit = b'{"kind":"event","value":42,"tags":["bcs","test"],"ok":true}\n'
        return (unit * ((size // len(unit)) + 2))[:size]
    raise ValueError(f"unknown mode: {mode}")


def time_pipeline(data: bytes, steps: int, token: bool, n_active_bytes: int) -> dict:
    timings = {}

    t0 = time.perf_counter()
    substrate = ByteSubstrate(data)
    timings["substrate_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    field = FieldSystemV6(substrate=substrate, n_active_bytes=n_active_bytes)
    timings["field_init_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    for _ in range(steps):
        field.step()
    timings["field_steps_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    organizer = SelfOrganizerV4(field_system=field)
    clusters = organizer.detect_clusters()
    timings["clusters_ms"] = (time.perf_counter() - t0) * 1000.0

    tokens = []
    if token:
        t0 = time.perf_counter()
        discoverer = EmergentTokenDiscovery(
            min_frequency=2,
            max_token_length=8,
            min_info_gain=0.01,
        )
        tokens = discoverer.discover(substrate, clusters)
        timings["tokens_ms"] = (time.perf_counter() - t0) * 1000.0
    else:
        timings["tokens_ms"] = 0.0

    total = sum(timings.values())
    timings.update(
        {
            "total_ms": total,
            "bytes": len(data),
            "kb": len(data) / 1024.0,
            "steps": steps,
            "clusters": len(clusters),
            "tokens": len(tokens),
            "ms_per_kb": total / max(len(data) / 1024.0, 1e-9),
        }
    )
    return timings


def summarize(rows: list[dict]) -> dict:
    keys = [
        "substrate_ms",
        "field_init_ms",
        "field_steps_ms",
        "clusters_ms",
        "tokens_ms",
        "total_ms",
        "ms_per_kb",
        "clusters",
        "tokens",
    ]
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in rows]
        out[key] = {
            "mean": mean(vals),
            "stdev": stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1024,4096,16384,32768")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mode", choices=["text", "binary", "structured"], default="text")
    parser.add_argument("--n-active-bytes", type=int, default=64)
    parser.add_argument("--no-token", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    all_results = []

    for size in sizes:
        data = make_data(size, args.mode)
        rows = []
        for _ in range(args.repeat):
            gc.collect()
            rows.append(
                time_pipeline(
                    data,
                    steps=args.steps,
                    token=not args.no_token,
                    n_active_bytes=args.n_active_bytes,
                )
            )
        summary = summarize(rows)
        result = {
            "size": size,
            "mode": args.mode,
            "steps": args.steps,
            "repeat": args.repeat,
            "n_active_bytes": args.n_active_bytes,
            "summary": summary,
        }
        all_results.append(result)

    if args.json:
        print(json.dumps(all_results, indent=2, ensure_ascii=False))
    else:
        print("size,steps,n_active,total_ms,ms_per_kb,substrate,field_init,field_steps,clusters,tokens,clusters_n,tokens_n")
        for result in all_results:
            s = result["summary"]
            print(
                f"{result['size']},{result['steps']},{result['n_active_bytes']},"
                f"{s['total_ms']['mean']:.3f},{s['ms_per_kb']['mean']:.3f},"
                f"{s['substrate_ms']['mean']:.3f},{s['field_init_ms']['mean']:.3f},"
                f"{s['field_steps_ms']['mean']:.3f},{s['clusters_ms']['mean']:.3f},"
                f"{s['tokens_ms']['mean']:.3f},{s['clusters']['mean']:.1f},{s['tokens']['mean']:.1f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
