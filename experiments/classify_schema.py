"""Does the MoEGL graph add anything a bag of names cannot see?

Same near-duplicate split and the same LinearSVC protocol as classify_v2.py
(stratified 5-fold x 2 seeds, C chosen on a 15% validation slice, test scored
once). The only change is the document built from the MoEGL graph.

  names          element-name tokens (the Lopez / WordE4MDE-style baseline)
  typed          the same tokens prefixed by the Ecore class (EClass, EReference, ...)
  schema         names plus one token per structural feature:
                 owner > attr|cont|ref : lower-upper > type
                 plus extends: and abstract:/interface: markers
  graph          schema plus one token per named edge (src > edgeType > dst)

Names use the same TfidfVectorizer settings as classify_v2.py. Graph tokens
use a whitespace token pattern so '>' stays inside one feature. Both
vectorizers are fit on the training fold only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

from classify_v2 import CFG, FOLDS, LABELS, SEEDS, val_split  # noqa: E402
from moegl import extract, load_config  # noqa: E402
from moegl.adapt import adapt  # noqa: E402
from moegl.encoders import tokenize  # noqa: E402

OUT = os.path.join(HERE, "results", "exp4_schema_types.json")


# External Ecore datatypes are written under several spellings
# (EString / String, EInt / EInteger). Collapsing them keeps one feature.
_TYPE_ALIAS = {
    "estring": "string", "string": "string",
    "eint": "int", "einteger": "int", "int": "int", "integer": "int",
    "eintegerobject": "int",
    "eboolean": "boolean", "boolean": "boolean", "ebooleanobject": "boolean",
    "edouble": "double", "double": "double", "edoubleobject": "double",
    "efloat": "float", "float": "float", "efloatobject": "float",
    "elong": "long", "long": "long", "elongobject": "long",
    "eshort": "short", "short": "short",
    "ebyte": "byte", "byte": "byte",
    "echar": "char", "char": "char",
    "edate": "date", "date": "date",
    "eobject": "eobject",
}


def phrase(node):
    toks = tokenize(node["attrs"].get("name", ""))
    return "_".join(toks)


def canonical_type(raw):
    toks = tokenize(raw)
    key = "".join(toks)
    return _TYPE_ALIAS.get(key, "_".join(toks))


def flag(value):
    return str(value).strip().lower() == "true"


def bound_token(value, default):
    if value is None:
        value = default
    s = str(value).strip()
    if s in ("*", "-1"):
        return "star"
    if s.lstrip("-").isdigit():
        return str(int(s))
    return "na"


def documents(models, external=True):
    """Return parallel document strings. Order follows ``models``."""
    names, typed, schema, graph = [], [], [], []
    n_schema = n_rel = 0
    for model in models:
        by_id = {n["id"]: n for n in model["nodes"]}
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        for edge in model["edges"]:
            outgoing[edge["src"]].append(edge)
            incoming[edge["dst"]].append(edge)

        name_toks = []
        typed_toks = []
        for node in model["nodes"]:
            toks = tokenize(node["attrs"].get("name", ""))
            name_toks.extend(toks)
            typed_toks.extend(f"{node['type']}:{tok}" for tok in toks)
            if flag(node["attrs"].get("abstract")) and phrase(node):
                typed_toks.append("abstract:" + phrase(node))
            if flag(node["attrs"].get("interface")) and phrase(node):
                typed_toks.append("interface:" + phrase(node))

        schema_toks = []
        for node in model["nodes"]:
            if node["type"] not in ("EAttribute", "EReference"):
                continue
            owner = type_name = None
            for edge in outgoing[node["id"]]:
                dst = by_id.get(edge["dst"])
                if dst is None:
                    continue
                p = phrase(dst)
                if not p:
                    continue
                if edge["type"] == "eContainingClass":
                    owner = p
                elif edge["type"] == "eType":
                    type_name = p
            if owner is None:
                for edge in incoming[node["id"]]:
                    if edge["type"] != "eStructuralFeatures":
                        continue
                    src = by_id.get(edge["src"])
                    if src is not None and phrase(src):
                        owner = phrase(src)
                        break
            if owner is None:
                continue
            if type_name is None and external:
                ext = node["attrs"].get("eTypeName")
                if ext:
                    type_name = canonical_type(ext) or None
            if node["type"] == "EAttribute":
                kind = "attr"
            elif flag(node["attrs"].get("containment")):
                kind = "cont"
            else:
                kind = "ref"
            lb = bound_token(node["attrs"].get("lowerBound"), "0")
            ub = bound_token(node["attrs"].get("upperBound"), "1")
            schema_toks.append(f"{owner}>{kind}:{lb}-{ub}>{type_name or 'untyped'}")
        for edge in model["edges"]:
            if edge["type"] != "eSuperTypes":
                continue
            src, dst = by_id.get(edge["src"]), by_id.get(edge["dst"])
            if src is None or dst is None or not phrase(src) or not phrase(dst):
                continue
            schema_toks.append(f"{phrase(src)}>extends>{phrase(dst)}")

        rel_toks = []
        for edge in model["edges"]:
            src, dst = by_id.get(edge["src"]), by_id.get(edge["dst"])
            if src is None or dst is None or not phrase(src) or not phrase(dst):
                continue
            rel_toks.append(f"{phrase(src)}>{edge['type']}>{phrase(dst)}")

        n_schema += len(schema_toks)
        n_rel += len(rel_toks)
        names.append(" ".join(name_toks))
        typed.append(" ".join(typed_toks))
        schema.append(" ".join(schema_toks))
        graph.append(" ".join(rel_toks))
    n_untyped = sum(doc.count(">untyped") for doc in schema)
    print(
        f"docs {len(names)} schema_tokens {n_schema} rel_tokens {n_rel} "
        f"untyped {n_untyped}",
        flush=True,
    )
    return names, typed, schema, graph


def fit_predict(name_docs, extra_docs, y, tr, va, te, c, max_extra=50000, dense=None, dense_weight=1.0):
    name_vec = TfidfVectorizer(min_df=2, max_features=20000)
    xtr = name_vec.fit_transform([name_docs[i] for i in tr])
    xva = name_vec.transform([name_docs[i] for i in va])
    xte = name_vec.transform([name_docs[i] for i in te])
    if extra_docs is not None:
        extra_vec = TfidfVectorizer(
            token_pattern=r"(?u)\S+", min_df=2, max_features=max_extra, lowercase=True,
        )
        etr = extra_vec.fit_transform([extra_docs[i] for i in tr])
        eva = extra_vec.transform([extra_docs[i] for i in va])
        ete = extra_vec.transform([extra_docs[i] for i in te])
        xtr = sparse.hstack([xtr, etr]).tocsr()
        xva = sparse.hstack([xva, eva]).tocsr()
        xte = sparse.hstack([xte, ete]).tocsr()
    if dense is not None:
        block = np.asarray(dense, dtype=np.float64) * float(dense_weight)
        xtr = sparse.hstack([xtr, sparse.csr_matrix(block[tr])]).tocsr()
        xva = sparse.hstack([xva, sparse.csr_matrix(block[va])]).tocsr()
        xte = sparse.hstack([xte, sparse.csr_matrix(block[te])]).tocsr()
    clf = LinearSVC(C=c, class_weight="balanced", max_iter=4000, dual=True)
    clf.fit(xtr, y[tr])
    return clf.predict(xva), clf.predict(xte)


def evaluate(name, name_docs, extra_docs, y, max_extra=50000, dense=None, weights=(1.0,)):
    accs, bals = [], []
    t0 = time.perf_counter()
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_all, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            tr, va = val_split(tr_all, y, seed + fold)
            best_c, best_bal, best_te = 1.0, -1.0, None
            for c in (0.1, 1.0, 10.0):
                for w in weights:
                    pred_va, pred_te = fit_predict(
                        name_docs, extra_docs, y, tr, va, te, c,
                        max_extra=max_extra, dense=dense, dense_weight=w,
                    )
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
        "n": len(accs),
        "bals": [float(b) for b in bals],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(
        f"SUMMARY {name}: acc {summary['acc_mean']:.3f}+-{summary['acc_sd']:.3f} "
        f"bal {summary['bal_mean']:.3f}+-{summary['bal_sd']:.3f}",
        flush=True,
    )
    return summary


def main():
    cfg = load_config(CFG)
    t0 = time.perf_counter()
    adapted = adapt(extract(cfg), cfg)
    print(f"extract+adapt {time.perf_counter() - t0:.1f}s", flush=True)
    labels = json.load(open(LABELS, encoding="utf-8"))["labels"]
    models = [
        m for m in adapted["models"]
        if m["name"] in labels and m["nodes"]
    ]
    models.sort(key=lambda m: m["name"])
    cats = sorted({labels[m["name"]] for m in models})
    c2i = {c: i for i, c in enumerate(cats)}
    y = np.array([c2i[labels[m["name"]]] for m in models])
    print(f"kept {len(models)} cats {len(cats)}", flush=True)
    name_docs, typed_docs, schema_docs, rel_docs = documents(models)
    # show that schema tokens are real relations, not a re-bag of names
    sample = schema_docs[0].split()[:8]
    print("sample schema:", sample, flush=True)

    results = {
        "n": len(models),
        "n_categories": len(cats),
        "protocol": "5-fold x 2 seeds, model selection on validation only, LinearSVC",
        "variants": {},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    specs = (
        ("names", name_docs, None),
        ("names+typed", name_docs, typed_docs),
        ("names+schema", name_docs, schema_docs),
        ("names+schema+edges", name_docs, [
            (s + " " + r).strip() for s, r in zip(schema_docs, rel_docs)
        ]),
    )
    for name, base, extra in specs:
        print(f"\n=== {name} ===", flush=True)
        results["variants"][name] = evaluate(name, base, extra, y)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    base_b = np.array(results["variants"]["names"]["bals"])
    for name, summary in results["variants"].items():
        if name == "names":
            continue
        delta = np.array(summary["bals"]) - base_b
        summary["delta_vs_names_mean"] = float(delta.mean())
        summary["delta_vs_names_sd"] = float(delta.std(ddof=1))
        print(
            f"DELTA {name} - names: {delta.mean():+.4f} +- {delta.std(ddof=1):.4f}",
            flush=True,
        )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
