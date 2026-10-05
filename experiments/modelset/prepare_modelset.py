"""Prepare the ModelSet Ecore dataset for the learning experiment (R1.4).

Follows the protocol of Lopez, Rubei, Sanchez Cuadrado and Di Ruscio,
"Machine learning methods for model classification: a comparative study"
(MODELS 2022):
  * labels: main ModelSet category; `dummy`, `unknown` and unlabelled models
    are discarded;
  * near-duplicates removed with the criterion of Allamanis (2019): two models
    are duplicates when the Jaccard similarity of their token sets is >= 0.8
    and the Jaccard similarity of their token multisets is >= 0.7; tokens are
    the ModelSet 1-gram text files; one representative (smallest id) is kept
    per duplicate cluster;
  * categories with fewer than 10 models are dropped (separately for the
    variant with and without duplicates).

Output (in --out):
  models/<n>.ecore      flat copy of every selected model (MoEGL input)
  labels_all.csv        name,category,modelset_id   (with duplicates)
  labels_nodup.csv      same, duplicates removed
  tokens.json           name -> ModelSet 1-gram tokens (for TF-IDF baselines)
  summary.json          dataset statistics

Usage: python prepare_modelset.py --modelset E:/Project/ModelSet/modelset --out E:/Project/ModelSet/prepared
"""
import argparse
import csv
import json
import os
import shutil
import sqlite3
from collections import Counter

MIN_PER_CATEGORY = 10
DROP = {"dummy", "unknown", None}


def long_path(p):
    p = os.path.abspath(p)
    return p if os.name != "nt" or p.startswith("\\\\?\\") else "\\\\?\\" + p


def read_tokens(txt_dir):
    toks = []
    for f in sorted(os.listdir(long_path(txt_dir))):
        with open(long_path(os.path.join(txt_dir, f)), encoding="utf-8", errors="replace") as fh:
            toks += [t.strip().lower() for t in fh if t.strip()]
    return toks


def jaccard_sets(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def jaccard_multisets(a, b):
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 1.0


def dedup(items):
    """items: list of (name, token list). Returns set of names to keep."""
    sets = {n: set(t) for n, t in items}
    bags = {n: Counter(t) for n, t in items}
    order = sorted(sets, key=lambda n: len(sets[n]))
    parent = {n: n for n in order}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(order):
        la = len(sets[a])
        for b in order[i + 1:]:
            if la < 0.8 * len(sets[b]):  # set Jaccard cannot reach 0.8
                break
            if jaccard_sets(sets[a], sets[b]) >= 0.8 and jaccard_multisets(bags[a], bags[b]) >= 0.7:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return {n for n in order if find(n) == n}


def filter_small(rows):
    counts = Counter(r["category"] for r in rows)
    return [r for r in rows if counts[r["category"]] >= MIN_PER_CATEGORY]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    db = os.path.join(args.modelset, "datasets", "dataset.ecore", "data", "ecore.db")
    raw_root = os.path.join(args.modelset, "raw-data", "repo-ecore-all")
    txt_root = os.path.join(args.modelset, "txt", "repo-ecore-all")
    rows = sqlite3.connect(db).execute(
        "select m.id, m.filename, md.json from models m join metadata md on m.id = md.id order by m.id"
    ).fetchall()

    selected, missing = [], 0
    for mid, filename, js in rows:
        cat = (json.loads(js).get("category") or [None])[0] if js else None
        if cat in DROP:
            continue
        raw = os.path.join(raw_root, filename)
        txt = os.path.join(txt_root, filename)
        if not os.path.exists(long_path(raw)) or not os.path.isdir(long_path(txt)):
            missing += 1
            continue
        selected.append({"modelset_id": mid, "category": cat, "raw": raw, "tokens": read_tokens(txt)})

    labelled = filter_small(selected)
    keep = dedup([(r["modelset_id"], r["tokens"]) for r in labelled])
    nodup = filter_small([r for r in labelled if r["modelset_id"] in keep])

    models_dir = os.path.join(args.out, "models")
    os.makedirs(models_dir, exist_ok=True)
    tokens = {}
    for i, r in enumerate(labelled):
        r["name"] = f"m{i:05d}"
        shutil.copyfile(long_path(r["raw"]), os.path.join(models_dir, r["name"] + ".ecore"))
        tokens[r["name"]] = r["tokens"]

    def write(path, rs):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "category", "modelset_id"])
            for r in rs:
                w.writerow([r["name"], r["category"], r["modelset_id"]])

    write(os.path.join(args.out, "labels_all.csv"), labelled)
    write(os.path.join(args.out, "labels_nodup.csv"), nodup)
    with open(os.path.join(args.out, "tokens.json"), "w", encoding="utf-8") as f:
        json.dump(tokens, f)
    summary = {
        "modelset_rows": len(rows),
        "missing_files": missing,
        "all": {"models": len(labelled), "categories": len({r["category"] for r in labelled})},
        "nodup": {"models": len(nodup), "categories": len({r["category"] for r in nodup})},
        "lopez2022_reported": {"all": [4167, 67], "nodup": [2068, 48]},
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
