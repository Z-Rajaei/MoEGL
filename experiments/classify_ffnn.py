"""Same split and MoEGL documents as the 82.4% linear model, but the
classifier is a one-hidden-layer network, the model class that won in
Lopez et al. MODELS 2022 (FFNN + TF-IDF, 82.5% without duplicates).

Hidden size is chosen on the validation slice from {100, 200}. The test
fold is scored once. Two variants: names only, and names plus MoEGL
relation tokens.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from classify_schema import CFG, LABELS, documents  # noqa: E402
from classify_v2 import FOLDS, SEEDS, val_split  # noqa: E402
from moegl import extract, load_config  # noqa: E402
from moegl.adapt import adapt  # noqa: E402

OUT = os.path.join(HERE, "results", "exp4_ffnn.json")
HIDDENS = (100, 200)
EPOCHS = 40
PATIENCE = 6
BATCH = 64
LR = 1e-3


class Net(torch.nn.Module):
    def __init__(self, d, h, c):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, h)
        self.fc2 = torch.nn.Linear(h, c)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def matrices(name_docs, extra_docs, tr, va, te):
    name_vec = TfidfVectorizer(min_df=2, max_features=20000)
    xtr = name_vec.fit_transform([name_docs[i] for i in tr])
    xva = name_vec.transform([name_docs[i] for i in va])
    xte = name_vec.transform([name_docs[i] for i in te])
    if extra_docs is not None:
        extra_vec = TfidfVectorizer(
            token_pattern=r"(?u)\S+", min_df=2, max_features=20000, lowercase=True,
        )
        xtr = sparse.hstack([xtr, extra_vec.fit_transform([extra_docs[i] for i in tr])]).tocsr()
        xva = sparse.hstack([xva, extra_vec.transform([extra_docs[i] for i in va])]).tocsr()
        xte = sparse.hstack([xte, extra_vec.transform([extra_docs[i] for i in te])]).tocsr()
    return xtr, xva, xte


def class_weight(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(w)


def predict(model, x):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, x.shape[0], 256):
            xb = torch.tensor(x[s:s + 256].toarray(), dtype=torch.float32)
            out.append(model(xb).argmax(dim=1).numpy())
    return np.concatenate(out)


def train_one(xtr, ytr, xva, yva, hidden, n_classes, seed):
    torch.manual_seed(seed)
    model = Net(xtr.shape[1], hidden, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    weight = class_weight(ytr, n_classes)
    rng = np.random.RandomState(seed)
    best_state, best_bal, wait = None, -1.0, 0
    for _ in range(EPOCHS):
        model.train()
        order = np.arange(xtr.shape[0])
        rng.shuffle(order)
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            xb = torch.tensor(xtr[b].toarray(), dtype=torch.float32)
            yb = torch.tensor(ytr[b], dtype=torch.long)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=weight)
            loss.backward()
            opt.step()
        bal = balanced_accuracy_score(yva, predict(model, xva))
        if bal > best_bal:
            best_bal = bal
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_bal


def evaluate(name, name_docs, extra_docs, y):
    n_classes = int(y.max()) + 1
    accs, bals = [], []
    t0 = time.perf_counter()
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_all, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            tr, va = val_split(tr_all, y, seed + fold)
            xtr, xva, xte = matrices(name_docs, extra_docs, tr, va, te)
            best_h, best_bal, best_model = None, -1.0, None
            for h in HIDDENS:
                model, bal = train_one(xtr, y[tr], xva, y[va], h, n_classes, seed + fold + h)
                if bal > best_bal:
                    best_h, best_bal, best_model = h, bal, model
            pred = predict(best_model, xte)
            accs.append(accuracy_score(y[te], pred))
            bals.append(balanced_accuracy_score(y[te], pred))
            print(
                f"  {name} seed {seed} fold {fold}: "
                f"acc={accs[-1]:.3f} bal={bals[-1]:.3f} h={best_h}",
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
    models = sorted(
        (m for m in adapted["models"] if m["name"] in labels and m["nodes"]),
        key=lambda m: m["name"],
    )
    cats = sorted({labels[m["name"]] for m in models})
    c2i = {c: i for i, c in enumerate(cats)}
    y = np.array([c2i[labels[m["name"]]] for m in models])
    name_docs, _, schema_docs, rel_docs = documents(models, external=False)
    extra = [(s + " " + r).strip() for s, r in zip(schema_docs, rel_docs)]
    results = {
        "n": len(models),
        "n_categories": len(cats),
        "protocol": "5-fold x 2 seeds, FFNN hidden in {100,200} chosen on validation, MoEGL relation tokens",
        "variants": {},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for name, ex in (("ffnn_names", None), ("ffnn_names+graph", extra)):
        print(f"\n=== {name} ===", flush=True)
        results["variants"][name] = evaluate(name, name_docs, ex, y)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    base = np.array(results["variants"]["ffnn_names"]["bals"])
    other = np.array(results["variants"]["ffnn_names+graph"]["bals"])
    delta = other - base
    results["delta_graph_minus_names"] = {
        "mean": float(delta.mean()),
        "sd": float(delta.std(ddof=1)),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(
        f"DELTA graph - names: {delta.mean():+.4f} +- {delta.std(ddof=1):.4f}",
        flush=True,
    )
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
