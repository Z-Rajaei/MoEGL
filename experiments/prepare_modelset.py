"""Flatten ModelSet Ecore models and write labels for Experiment 4.

Keeps categories with at least ``--min-per-class`` labelled models (default 10),
drops ``dummy`` / ``unknown``, and optionally collapses exact structural
duplicates (same node-type multiset + typed-edge multiset after a MoEGL
structure-only encoding). Writes:

  experiments/data/modelset_ecore/models/<id>.ecore
  experiments/data/modelset_ecore/labels.json
  experiments/data/modelset_ecore/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

DEFAULT_DB = r"E:\Project\ModelSet\modelset\datasets\dataset.ecore\data\ecore.db"
DEFAULT_RAW = r"E:\Project\ModelSet\modelset\raw-data\repo-ecore-all"
OUT = os.path.join(HERE, "data", "modelset_ecore")

DROP = {"dummy", "unknown", None}


def long(path):
    p = os.path.normpath(path)
    if p.startswith("\\\\?\\"):
        return p
    return "\\\\?\\" + p


def category(raw_json):
    if not raw_json:
        return None
    data = json.loads(raw_json)
    val = data.get("category")
    if isinstance(val, list):
        return val[0] if val else None
    return val


def collect_rows(db_path, raw_root, min_per_class):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT m.id, m.filename, md.json FROM models m JOIN metadata md ON m.id = md.id"
    ).fetchall()
    items = []
    missing = 0
    for mid, filename, j in rows:
        cat = category(j)
        src = long(os.path.join(raw_root, filename.replace("/", os.sep)))
        if not os.path.isfile(src):
            missing += 1
            continue
        items.append({"id": mid, "filename": filename, "category": cat, "src": src})
    counts = {}
    for it in items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    keep = {c for c, n in counts.items() if n >= min_per_class and c not in DROP}
    kept = [it for it in items if it["category"] in keep]
    print(
        f"db={len(rows)} found={len(items)} missing={missing} "
        f"cats>={min_per_class} excl dummy/unknown: {len(keep)} models={len(kept)}"
    )
    return kept, keep


def flatten(items, dest_models):
    os.makedirs(dest_models, exist_ok=True)
    mapping = []
    for i, it in enumerate(items):
        dest = os.path.join(dest_models, f"{i:04d}.ecore")
        if not os.path.isfile(dest):
            shutil.copy2(it["src"], dest)
        mapping.append({**it, "flat": dest, "flat_name": f"{i:04d}"})
        if (i + 1) % 500 == 0:
            print(f"  copied {i + 1}/{len(items)}")
    return mapping


def exact_duplicate_keys(mapping):
    """Cheap exact-duplicate key from file bytes (identical .ecore files)."""
    from collections import defaultdict
    import hashlib

    buckets = defaultdict(list)
    for it in mapping:
        h = hashlib.md5()
        with open(it["flat"], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        buckets[h.hexdigest()].append(it)
    survivors = []
    dropped = 0
    for group in buckets.values():
        survivors.append(group[0])
        dropped += len(group) - 1
    print(f"byte-identical duplicates dropped: {dropped}; remaining {len(survivors)}")
    return survivors


def write_outputs(mapping, keep):
    os.makedirs(OUT, exist_ok=True)
    labels = {it["flat_name"]: it["category"] for it in mapping}
    with open(os.path.join(OUT, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": len(mapping),
                "categories": sorted(keep),
                "n_categories": len(keep),
                "labels": labels,
            },
            f,
            indent=2,
        )
    with open(os.path.join(OUT, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["flat_name", "category", "id", "filename"])
        w.writeheader()
        for it in mapping:
            w.writerow(
                {
                    "flat_name": it["flat_name"],
                    "category": it["category"],
                    "id": it["id"],
                    "filename": it["filename"],
                }
            )
    print("wrote", OUT, "n=", len(mapping), "cats=", len(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--min-per-class", type=int, default=10)
    ap.add_argument("--drop-identical", action="store_true", default=True)
    args = ap.parse_args()

    items, keep = collect_rows(args.db, args.raw, args.min_per_class)
    mapping = flatten(items, os.path.join(OUT, "models"))
    if args.drop_identical:
        mapping = exact_duplicate_keys(mapping)
        # re-filter classes that fell below the threshold after de-duplication
        counts = {}
        for it in mapping:
            counts[it["category"]] = counts.get(it["category"], 0) + 1
        keep = {c for c, n in counts.items() if n >= args.min_per_class}
        before = len(mapping)
        mapping = [it for it in mapping if it["category"] in keep]
        print(
            f"after re-filter: dropped {before - len(mapping)} from small classes; "
            f"{len(keep)} cats, {len(mapping)} models"
        )
    write_outputs(mapping, keep)


if __name__ == "__main__":
    main()
