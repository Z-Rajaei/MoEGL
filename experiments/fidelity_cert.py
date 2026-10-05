"""Does the MoEGL YAML reproduce the published Lopez graphs?

Matching is by the exact typed-neighbourhood signature (not its hash).
A pair counts as reproduced only when colour refinement returns a node
bijection that preserves every typed edge. A shared signature without
such a bijection is reported separately and is not counted as a match.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

from moegl import build_graphs, extract, load_config  # noqa: E402
from run_experiments import DATASETS, CONFIGS, published_graph, refinement_mapping  # noqa: E402

OUT = os.path.join(HERE, "results", "fidelity_cert.json")


def signature(g):
    types = {n: d["type"] for n, d in g.nodes(data=True)}
    per_node = []
    for n in g.nodes:
        out_e = tuple(sorted((d["type"], types[v]) for _, v, d in g.out_edges(n, data=True)))
        in_e = tuple(sorted((d["type"], types[u]) for u, _, d in g.in_edges(n, data=True)))
        per_node.append((types[n], out_e, in_e))
    return tuple(sorted(per_node))


def one(ds_name):
    ds = DATASETS[ds_name]
    cfg = load_config(os.path.join(CONFIGS, ds["config"]))
    t0 = time.perf_counter()
    graphs = build_graphs(extract(cfg), cfg)
    print(f"{ds_name}: encoded {len(graphs)} in {time.perf_counter()-t0:.1f}s", flush=True)
    buckets = defaultdict(list)
    for name, g in graphs.items():
        buckets[signature(g)].append(name)
    pub_files = sorted(f for f in os.listdir(ds["published"]) if f.endswith(".json"))
    certified = signature_only = 0
    unmatched = []
    for i, pf in enumerate(pub_files):
        g = published_graph(os.path.join(ds["published"], pf))
        key = signature(g)
        bucket = buckets.get(key)
        if not bucket:
            unmatched.append(pf)
            continue
        # Several models can share a signature. Take the first one for which
        # refinement yields an edge-preserving bijection.
        mapping = None
        chosen = None
        for name in bucket:
            mapping = refinement_mapping(graphs[name], g)
            if mapping is not None:
                chosen = name
                break
        if chosen is None:
            signature_only += 1
            bucket.pop(0)
        else:
            certified += 1
            bucket.remove(chosen)
        if (i + 1) % 100 == 0:
            print(f"  {ds_name} {i+1}/{len(pub_files)} certified={certified}", flush=True)
    extra = sum(len(v) for v in buckets.values())
    result = {
        "published": len(pub_files),
        "moegl": len(graphs),
        "certified_isomorphic": certified,
        "same_signature_no_bijection": signature_only,
        "unmatched_published": unmatched,
        "unused_moegl_graphs": extra,
    }
    print(
        f"{ds_name}: certified {certified}/{len(pub_files)} "
        f"signature_only {signature_only} unmatched {len(unmatched)} "
        f"extra {result['unused_moegl_graphs']}",
        flush=True,
    )
    return result


def main():
    results = {}
    for name in ("ecore", "rds", "yakindu"):
        results[name] = one(name)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
