"""Same GIN / RGCN protocol as classify_v2, plus the validation curve.

Does not write exp4_v2.json. Selection is unchanged: the epoch with the
highest validation balanced accuracy is kept, with the same patience.

Usage: python experiments/classify_convergence.py
"""
import json
import os
import sys
import time

import numpy as np
from gensim.models import KeyedVectors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import classify_v2 as c  # noqa: E402

OUT = os.path.join(HERE, "results", "exp4_convergence.json")


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": int(len(a)),
        "values": [float(v) for v in a],
    }


def summarise(curves):
    # Mean validation curve over the epochs that every fold reached.
    shortest = min(len(r["val_bal"]) for r in curves)
    val_mean = [
        float(np.mean([r["val_bal"][i] for r in curves]))
        for i in range(shortest)
    ]
    return {
        "n_folds": len(curves),
        "best_epoch": mean_sd([r["best_epoch"] for r in curves]),
        "epochs_run": mean_sd([r["epochs_run"] for r in curves]),
        "val_bal_at_best": mean_sd([r["val_bal_at_best"] for r in curves]),
        "test_bal": mean_sd([r["test_bal"] for r in curves]),
        "test_acc": mean_sd([r["test_acc"] for r in curves]),
        "val_bal_mean_through_epoch": val_mean,
        "folds": curves,
    }


def main():
    print("loading WordE4MDE", flush=True)
    kv = KeyedVectors.load(c.KV_PATH, mmap="r")
    rows, n_rel = c.load_models(kv)
    y = np.array([r["yi"] for r in rows])
    n_classes = int(y.max()) + 1
    data_name = c.make_data(rows, rich=False)
    results = {
        "n": len(rows),
        "n_categories": n_classes,
        "n_edge_types": n_rel,
        "protocol": (
            "Same as classify_v2: 5-fold x seeds (0, 1), 15% validation, "
            "epoch chosen by validation balanced accuracy, patience 6, max 30. "
            "GIN ignores edge types; RGCN uses them. Node features are the "
            "WordE4MDE name vector in both."
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variants": {},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for name, kind in (("gin_e4mde", "gin"), ("rgcn_e4mde", "rgcn")):
        print(f"\n=== {name} ===", flush=True)
        curves = []
        for seed in c.SEEDS:
            c.run_gnn(data_name, y, n_classes, n_rel, kind, seed, curves=curves)
        results["variants"][name] = summarise(curves)
        s = results["variants"][name]
        print(
            f"  best_epoch {s['best_epoch']['mean']:.2f}+-{s['best_epoch']['sd']:.2f} "
            f"epochs_run {s['epochs_run']['mean']:.2f}+-{s['epochs_run']['sd']:.2f} "
            f"val {s['val_bal_at_best']['mean']:.3f} "
            f"test {s['test_bal']['mean']:.3f}+-{s['test_bal']['sd']:.3f}",
            flush=True,
        )
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
