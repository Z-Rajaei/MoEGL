"""Fast Ecore fidelity (1-WL fingerprint) + timing. No VF2, no Java re-run."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from moegl import load_config, extract, build_graphs  # noqa: E402

from run_experiments import (  # noqa: E402
    CONFIGS,
    DATASETS,
    JAVA_BIN,
    LOPEZ_CP,
    PY,
    fingerprint,
    published_graph,
    time_lopez,
    time_moegl,
    time_moegl_reencode,
    env_spec,
    dataset_stats,
    mean_sd,
)


def fingerprint_fidelity(graphs, published_dir):
    from collections import defaultdict

    buckets = defaultdict(list)
    for name, g in graphs.items():
        buckets[fingerprint(g)].append(name)
    pub = [f for f in os.listdir(published_dir) if f.endswith(".json")]
    matched, unmatched = 0, []
    for pf in pub:
        a = published_graph(os.path.join(published_dir, pf))
        key = fingerprint(a)
        if buckets.get(key):
            buckets[key].pop()
            matched += 1
        else:
            unmatched.append(pf)
    extra = sum(len(v) for v in buckets.values())
    return {
        "published": len(pub),
        "reproduced": matched,
        "unmatched_published": unmatched,
        "extra_moegl_graphs": extra,
        "method": "typed_1WL_fingerprint",
    }


def main():
    ds = DATASETS["ecore"]
    cfg_path = os.path.join(CONFIGS, ds["config"])
    cfg = load_config(cfg_path)
    t0 = time.perf_counter()
    cache = extract(cfg)
    graphs = build_graphs(cache, cfg)
    print(f"encode {time.perf_counter() - t0:.2f}s models={len(graphs)}", flush=True)
    fid = fingerprint_fidelity(graphs, ds["published"])
    print(
        f"fidelity MoEGL {fid['reproduced']}/{fid['published']} "
        f"extra={fid['extra_moegl_graphs']} unmatched={fid['unmatched_published']}",
        flush=True,
    )
    stats = dataset_stats(cache, graphs)
    print("stats", {k: stats[k] for k in stats if k != "skipped"}, flush=True)

    print("timing MoEGL 10 runs (in-process)...", flush=True)
    s1, s2, tot = [], [], []
    for i in range(10):
        t0 = time.perf_counter()
        cache_i = extract(cfg)
        t1 = time.perf_counter()
        build_graphs(cache_i, cfg)
        t2 = time.perf_counter()
        s1.append((t1 - t0) * 1000)
        s2.append((t2 - t1) * 1000)
        tot.append((t2 - t0) * 1000)
        print(f"  run {i+1}: s1={s1[-1]:.0f} s2={s2[-1]:.0f}", flush=True)
    moegl = {
        "stage1_extract_ms": mean_sd(s1),
        "stage2_encode_ms": mean_sd(s2),
        "stage1_plus_stage2_ms": mean_sd(tot),
        "process_ms": mean_sd(tot),
        "note": "in-process; no interpreter start-up",
    }
    print(
        "MoEGL s1 %.0f+-%.0f s2 %.0f+-%.0f proc %.0f+-%.0f"
        % (
            moegl["stage1_extract_ms"]["mean"],
            moegl["stage1_extract_ms"]["sd"],
            moegl["stage2_encode_ms"]["mean"],
            moegl["stage2_encode_ms"]["sd"],
            moegl["process_ms"]["mean"],
            moegl["process_ms"]["sd"],
        ),
        flush=True,
    )
    print("timing Lopez Java 10 runs...", flush=True)
    try:
        lopez = time_lopez(10, "ecore")
        print(
            "Lopez enc %.0f+-%.0f proc %.0f+-%.0f"
            % (
                lopez["encode_ms"]["mean"],
                lopez["encode_ms"]["sd"],
                lopez["process_ms"]["mean"],
                lopez["process_ms"]["sd"],
            ),
            flush=True,
        )
    except Exception as exc:
        lopez = {"error": str(exc)}
        print("Lopez timing failed:", exc, flush=True)

    reenc = time_moegl_reencode(
        10, cache, os.path.join(CONFIGS, "exp1_alt_no_datatypes.yaml")
    )
    print("re-encode %.0f+-%.0f ms" % (reenc["mean"], reenc["sd"]), flush=True)

    out = {
        "env": env_spec(),
        "exp1_ecore": {
            "stats": {k: v for k, v in stats.items() if k != "skipped"},
            "fidelity_moegl_vs_published": fid,
        },
        "timing": {"ecore": {"lopez_java": lopez, "moegl": moegl},
                   "moegl_reencode_ecore_alt_config_ms": reenc},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(HERE, "results", "exp_results_fast.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("written", path, flush=True)


if __name__ == "__main__":
    main()
