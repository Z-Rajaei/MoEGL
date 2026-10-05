"""Can MoEGL structural statistics improve the WordE4MDE document vector?

Same near-duplicate split and validation protocol as classify_v2.py.
The baseline is the document-mean WordE4MDE vector (the published encoding).
The MoEGL vector appends normalised node-type and edge-type histograms and
the mean of structural attributes (containment, multiplicity, abstract, ...).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from classify_v2 import (  # noqa: E402
    FOLDS, KV_PATH, LABELS, SEEDS, load_models, val_split,
)
from gensim.models import KeyedVectors  # noqa: E402


def pack(rows):
    node_types = sorted({n.get("type", "?") for r in rows for n in r["nodes_raw"]})
    # rows from load_models don't keep type. Rebuild histograms inside load below.
    return node_types


def main():
    kv = KeyedVectors.load(KV_PATH, mmap="r")
    # reuse extraction but we need node types; load_models drops them.
    # Re-read labels and adapted cache the same way, keeping type.
    from moegl import load_config, extract
    from moegl.adapt import adapt
    from moegl.encoders import tokenize
    from classify_v2 import CFG, mean_vec, numeric_of

    cfg = load_config(CFG)
    adapted = adapt(extract(cfg), cfg)
    labels = json.load(open(LABELS, encoding="utf-8"))["labels"]
    node_types, edge_types = set(), set()
    raw = []
    for model in adapted["models"]:
        if model["name"] not in labels or not model["nodes"]:
            continue
        for n in model["nodes"]:
            node_types.add(n["type"])
        for e in model["edges"]:
            edge_types.add(e["type"])
        raw.append(model)
    node_types = sorted(node_types)
    edge_types = sorted(edge_types)
    nt = {t: i for i, t in enumerate(node_types)}
    et = {t: i for i, t in enumerate(edge_types)}
    docs, stats, y = [], [], []
    cats = sorted({labels[m["name"]] for m in raw})
    c2i = {c: i for i, c in enumerate(cats)}
    nd = len(numeric_of({}))
    for model in raw:
        toks = []
        hist_n = np.zeros(len(node_types), dtype=np.float32)
        hist_e = np.zeros(len(edge_types), dtype=np.float32)
        nums = []
        for n in model["nodes"]:
            toks.extend(tokenize(n["attrs"].get("name", "")))
            hist_n[nt[n["type"]]] += 1
            nums.append(numeric_of(n["attrs"]))
        for e in model["edges"]:
            hist_e[et[e["type"]]] += 1
        if hist_n.sum():
            hist_n /= hist_n.sum()
        if hist_e.sum():
            hist_e /= hist_e.sum()
        num = np.mean(nums, axis=0) if nums else np.zeros(nd, dtype=np.float32)
        docs.append(mean_vec(toks, kv))
        stats.append(np.concatenate([hist_n, hist_e, num]).astype(np.float32))
        y.append(c2i[labels[model["name"]]])
    y = np.array(y)
    docs = np.vstack(docs)
    stats = np.vstack(stats)
    both = np.concatenate([docs, stats], axis=1)
    print(f"n={len(y)} cats={len(cats)} doc={docs.shape[1]} stats={stats.shape[1]}", flush=True)

    def eval_matrix(name, x, kernel):
        accs, bals = [], []
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
            for fold, (tr_all, te) in enumerate(skf.split(np.zeros(len(y)), y)):
                tr, va = val_split(tr_all, y, seed + fold)
                scaler = StandardScaler()
                xtr = scaler.fit_transform(x[tr])
                xva = scaler.transform(x[va])
                xte = scaler.transform(x[te])
                best, best_bal, best_c = None, -1.0, None
                grid = (0.1, 1.0, 10.0) if kernel == "linear" else (1.0, 10.0)
                for c in grid:
                    if kernel == "linear":
                        clf = LinearSVC(C=c, class_weight="balanced", max_iter=5000, dual=False)
                    else:
                        clf = SVC(C=c, kernel="rbf", class_weight="balanced")
                    clf.fit(xtr, y[tr])
                    bal = balanced_accuracy_score(y[va], clf.predict(xva))
                    if bal > best_bal:
                        best, best_bal, best_c = clf, bal, c
                pred = best.predict(xte)
                accs.append(accuracy_score(y[te], pred))
                bals.append(balanced_accuracy_score(y[te], pred))
        print(
            f"{name:22s} acc {np.mean(accs):.3f}+-{np.std(accs, ddof=1):.3f} "
            f"bal {np.mean(bals):.3f}+-{np.std(bals, ddof=1):.3f}",
            flush=True,
        )
        return float(np.mean(bals)), float(np.std(bals, ddof=1))

    out = {}
    for kernel in ("linear", "rbf"):
        for name, mat in (("e4mde", docs), ("stats", stats), ("e4mde+structure", both)):
            bal, sd = eval_matrix(f"{kernel} {name}", mat, kernel)
            out[f"{kernel}:{name}"] = {"bal_mean": bal, "bal_sd": sd}
    path = os.path.join(HERE, "results", "exp4_stats.json")
    json.dump({"n": int(len(y)), "cats": len(cats), "results": out}, open(path, "w"), indent=2)
    print("written", path, flush=True)


if __name__ == "__main__":
    main()
