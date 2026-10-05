"""Collapse ModelSet copies that share the same multiset of `name="..."` tokens.

This is a conservative exact-name duplicate filter (not the Allamanis
near-duplicate detector of López et al. 2022). Categories that fall below
10 models afterwards are dropped. Writes labels_dedup.json next to labels.json.
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "data", "modelset_ecore")
NAME_RE = re.compile(r'name\s*=\s*"([^"]+)"')


def names_key(path):
    text = open(path, encoding="utf-8", errors="ignore").read()
    names = tuple(sorted(NAME_RE.findall(text)))
    return names


def main():
    with open(os.path.join(ROOT, "labels.json"), encoding="utf-8") as f:
        pack = json.load(f)
    labels = pack["labels"]
    buckets = defaultdict(list)
    models = os.path.join(ROOT, "models")
    for name, cat in labels.items():
        key = names_key(os.path.join(models, name + ".ecore"))
        buckets[key].append((name, cat))
    survivors = [group[0] for group in buckets.values()]
    dropped = len(labels) - len(survivors)
    counts = Counter(cat for _, cat in survivors)
    keep = {c for c, n in counts.items() if n >= 10}
    kept = [(n, c) for n, c in survivors if c in keep]
    out = {
        "n": len(kept),
        "categories": sorted(keep),
        "n_categories": len(keep),
        "labels": {n: c for n, c in kept},
        "dropped_name_duplicates": dropped,
        "dropped_small_classes": len(survivors) - len(kept),
        "note": "exact name-multiset duplicates; min 10 per class",
    }
    path = os.path.join(ROOT, "labels_dedup.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(ROOT, "manifest_dedup.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["flat_name", "category"])
        w.writerows(kept)
    print(
        f"all={len(labels)} name-dups-dropped={dropped} "
        f"after-min10={len(kept)} cats={len(keep)} -> {path}"
    )
    print("top cats", counts.most_common(10))


if __name__ == "__main__":
    main()
