"""Experiment 5 -- second MDE task: real vs synthetic Ecore (TCRMG-GNN).

Uses the same Lopez-style MoEGL config as Experiment 1, plus a name-word2vec
variant, to test whether encoding choices help a GNN tell real metamodels
from RAND synthetic ones. Protocol matches classify_gnn.py (5-fold, 3 seeds).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from classify_gnn import (  # noqa: E402
    SEEDS,
    build_dataset,
    summarise,
    train_eval_gnn,
    train_eval_tfidf,
    names_from_cache,
)
from moegl import load_config, extract, build_graphs  # noqa: E402

TCRMG = r"E:/Project/TCRMG-GNN-1.0.0"
REAL = os.path.join(TCRMG, "realModels", "Ecore")
SYN = os.path.join(TCRMG, "syntheticModels", "RAND", "Ecore")
OUT = os.path.join(HERE, "results", "exp5_tcrmg.json")
CFG_STRUCT = os.path.join(PKG, "configs", "exp1_lopez_ecore.yaml")
CFG_W2V = os.path.join(PKG, "configs", "exp5_tcrmg_name_w2v.yaml")


def encode(cfg_path, modelspath):
    cfg = load_config(cfg_path)
    cfg.modelspath = modelspath
    cache = extract(cfg)
    graphs = build_graphs(cache, cfg)
    return cache, graphs


def merge(real_g, syn_g):
    labels, graphs = {}, {}
    for name, g in real_g.items():
        key = "R_" + name
        graphs[key] = g
        labels[key] = "real"
    for name, g in syn_g.items():
        key = "S_" + name
        graphs[key] = g
        labels[key] = "synth"
    return graphs, labels


def run_variant(name, graphs, labels, cache, hetero, use_name, tfidf=False):
    print(f"\n=== {name} ===", flush=True)
    if tfidf:
        docs = names_from_cache(cache)
        names = [n for n in sorted(docs) if n in labels]
        y_idx = [0 if labels[n] == "real" else 1 for n in names]
        all_acc, all_bal = [], []
        for seed in SEEDS:
            a, b = train_eval_tfidf(docs, names, y_idx, 2, seed)
            all_acc += a
            all_bal += b
        return summarise(all_acc, all_bal)
    dataset, names, ys, _ = build_dataset(graphs, labels, use_name, hetero)
    all_acc, all_bal = [], []
    for seed in SEEDS:
        a, b = train_eval_gnn(dataset, ys, 2, hetero, seed)
        all_acc += a
        all_bal += b
    return summarise(all_acc, all_bal)


def main():
    print("encoding real + RAND synthetic Ecore", flush=True)
    t0 = time.perf_counter()
    real_c, real_g = encode(CFG_STRUCT, REAL)
    syn_c, syn_g = encode(CFG_STRUCT, SYN)
    graphs, labels = merge(real_g, syn_g)
    print(
        f"struct graphs real={len(real_g)} syn={len(syn_g)} "
        f"in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    results = {
        "task": "real_vs_synthetic_ecore",
        "n_real": len(real_g),
        "n_synth": len(syn_g),
        "variants": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results["variants"]["struct"] = run_variant(
        "struct", graphs, labels, real_c, hetero=False, use_name=False
    )

    real_c2, real_g2 = encode(CFG_W2V, REAL)
    syn_c2, syn_g2 = encode(CFG_W2V, SYN)
    # merge caches for tf-idf names
    cache = {
        "models": [
            {**m, "name": "R_" + m["name"]} for m in real_c2["models"]
        ]
        + [{**m, "name": "S_" + m["name"]} for m in syn_c2["models"]],
        "meta": {},
    }
    graphs2, labels2 = merge(real_g2, syn_g2)
    results["variants"]["name_w2v"] = run_variant(
        "name_w2v", graphs2, labels2, cache, hetero=False, use_name=True
    )
    results["variants"]["hetero_w2v"] = run_variant(
        "hetero_w2v", graphs2, labels2, cache, hetero=True, use_name=True
    )
    results["variants"]["tfidf_svm"] = run_variant(
        "tfidf_svm", graphs2, labels2, cache, hetero=False, use_name=True, tfidf=True
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    for k, v in results["variants"].items():
        print(f"SUMMARY {k}: acc {v['acc_mean']:.3f}+-{v['acc_sd']:.3f} "
              f"bal {v['bal_mean']:.3f}+-{v['bal_sd']:.3f}", flush=True)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
