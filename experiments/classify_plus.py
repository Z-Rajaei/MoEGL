"""Names + MoEGL graph tokens, with and without the WordE4MDE document vector.

Uses the relational tokens that beat bag-of-names (external datatypes left
untyped, 20k extra features). The document vector is the mean of WordE4MDE
over the model's name tokens, L2-normalised so it sits on the same scale as
TF-IDF. Its weight is chosen on the validation slice together with C.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from gensim.models import KeyedVectors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from classify_schema import (  # noqa: E402
    CFG, LABELS, documents, evaluate,
)
from classify_v2 import KV_PATH, mean_vec  # noqa: E402
from moegl import extract, load_config  # noqa: E402
from moegl.adapt import adapt  # noqa: E402
from moegl.encoders import tokenize  # noqa: E402

DEST = os.path.join(HERE, "results", "exp4_plus.json")


def doc_matrix(models, kv):
    rows = []
    for model in models:
        toks = []
        for node in model["nodes"]:
            toks.extend(tokenize(node["attrs"].get("name", "")))
        vec = mean_vec(toks, kv).astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        rows.append(vec)
    return np.vstack(rows)


def main():
    cfg = load_config(CFG)
    adapted = adapt(extract(cfg), cfg)
    labels = json.load(open(LABELS, encoding="utf-8"))["labels"]
    models = sorted(
        (m for m in adapted["models"] if m["name"] in labels and m["nodes"]),
        key=lambda m: m["name"],
    )
    cats = sorted({labels[m["name"]] for m in models})
    c2i = {c: i for i, c in enumerate(cats)}
    y = np.array([c2i[labels[m["name"]]] for m in models])
    name_docs, _, schema_docs, rel_docs = documents(models, external=False)
    extra = [(s + " " + r).strip() for s, r in zip(schema_docs, rel_docs)]
    print("loading WordE4MDE", flush=True)
    kv = KeyedVectors.load(KV_PATH, mmap="r")
    dense = doc_matrix(models, kv)
    results = {
        "n": len(models),
        "n_categories": len(cats),
        "protocol": "5-fold x 2 seeds, val-only C and embedding weight, external types off, max_extra 20000",
        "variants": {},
    }
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print("\n=== names+graph ===", flush=True)
    results["variants"]["names+graph"] = evaluate(
        "names+graph", name_docs, extra, y, max_extra=20000,
    )
    print("\n=== names+graph+e4mde ===", flush=True)
    results["variants"]["names+graph+e4mde"] = evaluate(
        "names+graph+e4mde", name_docs, extra, y, max_extra=20000,
        dense=dense, weights=(0.25, 1.0),
    )
    prev_path = os.path.join(HERE, "results", "exp4_schema.json")
    prev = json.load(open(prev_path, encoding="utf-8"))
    # names baseline is identical across runs; the typed file still has it
    base = np.array(prev["variants"]["names"]["bals"])
    for name, summary in results["variants"].items():
        delta = np.array(summary["bals"]) - base
        summary["delta_vs_names_mean"] = float(delta.mean())
        summary["delta_vs_names_sd"] = float(delta.std(ddof=1))
        print(f"DELTA {name} - names: {delta.mean():+.4f} +- {delta.std(ddof=1):.4f}", flush=True)
    with open(DEST, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("written", DEST, flush=True)


if __name__ == "__main__":
    main()
