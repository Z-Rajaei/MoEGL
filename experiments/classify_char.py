"""Character n-grams of identifiers, with and without MoEGL relation tokens.

Word TF-IDF is what Lopez et al. used (82.5% with a neural net). Character
n-grams keep fragments of camelCase names that a word split throws away.
The relation tokens stay whole (whitespace pattern), because splitting
'place>cont:0-star>transition' into characters would destroy the relation.
Same split, same LinearSVC protocol as the 82.4% run.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from classify_schema import CFG, LABELS, documents, evaluate  # noqa: E402
from classify_v2 import FOLDS, SEEDS, val_split  # noqa: E402
from moegl import extract, load_config  # noqa: E402
from moegl.adapt import adapt  # noqa: E402

OUT = os.path.join(HERE, "results", "exp4_char.json")


def fit_predict(name_docs, extra_docs, y, tr, va, te, c):
    name_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=50000, sublinear_tf=True,
    )
    xtr = name_vec.fit_transform([name_docs[i] for i in tr])
    xva = name_vec.transform([name_docs[i] for i in va])
    xte = name_vec.transform([name_docs[i] for i in te])
    if extra_docs is not None:
        extra_vec = TfidfVectorizer(
            token_pattern=r"(?u)\S+", min_df=2, max_features=20000,
            lowercase=True, sublinear_tf=True,
        )
        xtr = sparse.hstack([xtr, extra_vec.fit_transform([extra_docs[i] for i in tr])]).tocsr()
        xva = sparse.hstack([xva, extra_vec.transform([extra_docs[i] for i in va])]).tocsr()
        xte = sparse.hstack([xte, extra_vec.transform([extra_docs[i] for i in te])]).tocsr()
    clf = LinearSVC(C=c, class_weight="balanced", max_iter=4000, dual=True)
    clf.fit(xtr, y[tr])
    return clf.predict(xva), clf.predict(xte)


def run(name, name_docs, extra_docs, y):
    # local copy of the protocol so the vectorizer stays character-level
    accs, bals = [], []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_all, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            tr, va = val_split(tr_all, y, seed + fold)
            best_c, best_bal, best_te = 1.0, -1.0, None
            for c in (0.1, 1.0, 10.0):
                pred_va, pred_te = fit_predict(name_docs, extra_docs, y, tr, va, te, c)
                bal = balanced_accuracy_score(y[va], pred_va)
                if bal > best_bal:
                    best_bal, best_c, best_te = bal, c, pred_te
            accs.append(accuracy_score(y[te], best_te))
            bals.append(balanced_accuracy_score(y[te], best_te))
            print(
                f"  {name} seed {seed} fold {fold}: "
                f"acc={accs[-1]:.3f} bal={bals[-1]:.3f} C={best_c}",
                flush=True,
            )
    summary = {
        "acc_mean": float(np.mean(accs)),
        "acc_sd": float(np.std(accs, ddof=1)),
        "bal_mean": float(np.mean(bals)),
        "bal_sd": float(np.std(bals, ddof=1)),
        "bals": [float(b) for b in bals],
    }
    print(
        f"SUMMARY {name}: acc {summary['acc_mean']:.3f}+-{summary['acc_sd']:.3f} "
        f"bal {summary['bal_mean']:.3f}+-{summary['bal_sd']:.3f}",
        flush=True,
    )
    return summary


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
    results = {"n": len(models), "n_categories": len(cats), "variants": {}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for name, ex in (("char_names", None), ("char_names+graph", extra)):
        print(f"\n=== {name} ===", flush=True)
        results["variants"][name] = run(name, name_docs, ex, y)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    base = np.array(results["variants"]["char_names"]["bals"])
    other = np.array(results["variants"]["char_names+graph"]["bals"])
    delta = other - base
    print(f"DELTA graph - char names: {delta.mean():+.4f} +- {delta.std(ddof=1):.4f}", flush=True)
    results["delta"] = {"mean": float(delta.mean()), "sd": float(delta.std(ddof=1))}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
