"""Near-duplicate filter for ModelSet, adapted from Allamanis (2019).

Two models in the same category are near-duplicates when the Jaccard
similarity of their identifier-token sets is at least 0.8. Within each
category a greedy pass keeps the larger model. Categories left with fewer
than 10 models are dropped. Writes labels_neardup.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from moegl.encoders import tokenize  # noqa: E402

ROOT = os.path.join(HERE, "data", "modelset_ecore")
NAME_RE = re.compile(r'name\s*=\s*"([^"]+)"')
THRESH = 0.8


def token_set(path):
    text = open(path, encoding="utf-8", errors="ignore").read()
    toks = []
    for name in NAME_RE.findall(text):
        toks.extend(tokenize(name))
    return frozenset(toks)


def jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def main():
    with open(os.path.join(ROOT, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    models = os.path.join(ROOT, "models")
    by_cat = defaultdict(list)
    for name, cat in labels.items():
        toks = token_set(os.path.join(models, name + ".ecore"))
        by_cat[cat].append((name, toks))
    kept = []
    dropped = 0
    for cat, group in by_cat.items():
        group.sort(key=lambda it: len(it[1]), reverse=True)
        chosen = []
        for name, toks in group:
            if any(jaccard(toks, prev) >= THRESH for _, prev in chosen):
                dropped += 1
                continue
            chosen.append((name, toks))
        for name, _ in chosen:
            kept.append((name, cat))
    from collections import Counter
    counts = Counter(c for _, c in kept)
    keep_cats = {c for c, n in counts.items() if n >= 10}
    final = [(n, c) for n, c in kept if c in keep_cats]
    out = {
        "n": len(final),
        "n_categories": len(keep_cats),
        "categories": sorted(keep_cats),
        "labels": {n: c for n, c in final},
        "jaccard_threshold": THRESH,
        "dropped_neardup": dropped,
        "note": "within-category Jaccard >= 0.8 on identifier tokens; min 10 per class",
    }
    path = os.path.join(ROOT, "labels_neardup.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(
        f"start={len(labels)} neardup_dropped={dropped} "
        f"after_min10={len(final)} cats={len(keep_cats)} -> {path}"
    )


if __name__ == "__main__":
    main()
