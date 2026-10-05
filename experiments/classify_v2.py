"""Fair ModelSet experiment (near-duplicate split, no test-set model selection).

Split: labels_neardup.json (Jaccard >= 0.8, min 10/class; ~2074 models, 48
categories -- the same size as Lopez et al. MODELS 2022, Table 2, no-dups).

Protocol: stratified 5-fold x 2 seeds. Inside each training fold, 15% is a
validation set. GNN epoch and SVM C are chosen on validation only; the test
fold is scored once.

Variants
  e4mde_svm   document-mean WordE4MDE + LinearSVC (WordE4MDE paper protocol)
  tfidf_svm   TF-IDF of the same tokens + LinearSVC
  gin_e4mde   homogeneous GIN, node = WordE4MDE(name)
  rgcn_e4mde  heterogeneous RGCN, same node features, typed edges
  rgcn_rich   RGCN + structural attributes (multiplicity, containment, abstract)
  fusion      RGCN-rich readout concatenated with the document WordE4MDE vector
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, RGCNConv, global_mean_pool

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from moegl import load_config, extract  # noqa: E402
from moegl.adapt import adapt  # noqa: E402
from moegl.encoders import tokenize, to_float  # noqa: E402

LABELS = os.path.join(HERE, "data", "modelset_ecore", "labels_neardup.json")
CFG = os.path.join(PKG, "configs", "exp4_modelset_rich.yaml")
KV_PATH = r"E:\Project\ModelSet\worde4mde\out\skip_gram_modelling\skip_gram_vectors.kv"
OUT = os.path.join(HERE, "results", "exp4_v2.json")

HIDDEN = 128
LAYERS = 2
EPOCHS = 30
PATIENCE = 6
LR = 1e-3
BATCH = 64
SEEDS = (0, 1)
FOLDS = 5
NUM_KEYS = (
    "abstract", "interface", "containment", "derived",
    "ordered", "unique", "changeable", "iD", "lowerBound",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def numeric_of(attrs):
    vec = []
    for key in NUM_KEYS:
        raw = attrs.get(key)
        vec.append(0.0 if raw is None else to_float(raw))
    upper = attrs.get("upperBound")
    if upper is None:
        vec.append(0.0)
    else:
        s = str(upper).strip()
        vec.append(1.0 if s in ("*", "-1") or (s.lstrip("-").isdigit() and int(float(s)) < 0) else 0.0)
    return vec


def mean_vec(tokens, kv):
    vecs = [kv[t] for t in tokens if t in kv]
    if not vecs:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def load_models(kv):
    cfg = load_config(CFG)
    t0 = time.perf_counter()
    adapted = adapt(extract(cfg), cfg)
    print(f"extract+adapt {time.perf_counter()-t0:.1f}s models={len(adapted['models'])}", flush=True)
    with open(LABELS, encoding="utf-8") as f:
        labels = json.load(f)
    lab = labels["labels"]
    edge_types = sorted({
        e["type"] for m in adapted["models"] for e in m["edges"]
    })
    e2i = {t: i for i, t in enumerate(edge_types)}
    rows = []
    for model in adapted["models"]:
        if model["name"] not in lab or not model["nodes"]:
            continue
        doc_toks = []
        nodes = []
        for node in model["nodes"]:
            toks = tokenize(node["attrs"].get("name", ""))
            doc_toks.extend(toks)
            nodes.append({
                "vec": mean_vec(toks, kv),
                "num": numeric_of(node["attrs"]),
                "id": node["id"],
            })
        loc = {n["id"]: i for i, n in enumerate(nodes)}
        edges = []
        for e in model["edges"]:
            if e["src"] in loc and e["dst"] in loc:
                edges.append((loc[e["src"]], loc[e["dst"]], e2i[e["type"]]))
        rows.append({
            "y": lab[model["name"]],
            "text": " ".join(doc_toks),
            "doc": mean_vec(doc_toks, kv),
            "nodes": nodes,
            "edges": edges,
        })
    cats = sorted({r["y"] for r in rows})
    c2i = {c: i for i, c in enumerate(cats)}
    for r in rows:
        r["yi"] = c2i[r["y"]]
    print(
        f"kept {len(rows)} cats {len(cats)} edge_types {len(edge_types)} "
        f"e4dim {kv.vector_size}",
        flush=True,
    )
    return rows, len(edge_types)


def make_data(rows, rich):
    dim = rows[0]["nodes"][0]["vec"].shape[0]
    nd = len(rows[0]["nodes"][0]["num"])
    out = []
    for r in rows:
        n = len(r["nodes"])
        x = np.zeros((n, dim + (nd if rich else 0)), dtype=np.float32)
        for i, node in enumerate(r["nodes"]):
            x[i, :dim] = node["vec"]
            if rich:
                x[i, dim:] = node["num"]
        if r["edges"]:
            src, dst, et = zip(*r["edges"])
        else:
            src, dst, et = (0,), (0,), (0,)
        out.append(Data(
            x=torch.tensor(x),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_type=torch.tensor(et, dtype=torch.long),
            y=torch.tensor([r["yi"]], dtype=torch.long),
            bow=torch.tensor(r["doc"]).view(1, -1),
            num_nodes=n,
        ))
    return out


class GINNet(torch.nn.Module):
    def __init__(self, in_dim, n_classes, extra=0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        last = in_dim
        for _ in range(LAYERS):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(last, HIDDEN), torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN, HIDDEN),
            )
            self.convs.append(GINConv(mlp))
            last = HIDDEN
        self.lin = torch.nn.Linear(HIDDEN + extra, n_classes)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, ei))
        h = global_mean_pool(x, data.batch)
        return self.lin(h)


class RGCNNet(torch.nn.Module):
    def __init__(self, in_dim, n_classes, n_rel, extra=0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        last = in_dim
        for _ in range(LAYERS):
            self.convs.append(RGCNConv(last, HIDDEN, num_relations=max(n_rel, 1)))
            last = HIDDEN
        self.lin = torch.nn.Linear(HIDDEN + extra, n_classes)
        self.extra = extra

    def forward(self, data):
        x, ei, et = data.x, data.edge_index, data.edge_type
        for conv in self.convs:
            x = F.relu(conv(x, ei, et))
        h = global_mean_pool(x, data.batch)
        if self.extra:
            h = torch.cat([h, data.bow], dim=1)
        return self.lin(h)


def val_split(train_idx, y, seed):
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for c in np.unique(y[train_idx]):
        idx = train_idx[y[train_idx] == c].copy()
        rng.shuffle(idx)
        n_va = int(round(len(idx) * 0.15)) if len(idx) >= 5 else 0
        va.extend(idx[:n_va])
        tr.extend(idx[n_va:])
    return np.array(tr), np.array(va)


def class_weight(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(w)


def run_gnn(dataset, y, n_classes, n_rel, kind, seed, curves=None):
    set_seed(seed)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    accs, bals = [], []
    in_dim = dataset[0].x.size(1)
    extra = dataset[0].bow.numel() if kind == "fusion" else 0
    for fold, (tr_all, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr, va = val_split(tr_all, y, seed + fold)
        if kind == "gin":
            model = GINNet(in_dim, n_classes)
        elif kind == "fusion":
            model = RGCNNet(in_dim, n_classes, n_rel, extra=extra)
        else:
            model = RGCNNet(in_dim, n_classes, n_rel)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        weight = class_weight(y[tr], n_classes)
        tr_loader = DataLoader([dataset[i] for i in tr], batch_size=BATCH, shuffle=True)
        va_loader = DataLoader([dataset[i] for i in va], batch_size=BATCH)
        te_loader = DataLoader([dataset[i] for i in te], batch_size=BATCH)
        best_state, best_bal, wait = None, -1.0, 0
        best_epoch = 0
        val_bal, train_loss = [], []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss, n_batches = 0.0, 0
            for batch in tr_loader:
                opt.zero_grad()
                loss = F.cross_entropy(model(batch), batch.y, weight=weight)
                loss.backward()
                opt.step()
                total_loss += float(loss.detach())
                n_batches += 1
            bal = score(model, va_loader)[1]
            val_bal.append(bal)
            train_loss.append(total_loss / max(n_batches, 1))
            if bal > best_bal:
                best_bal = bal
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        acc, bal = score(model, te_loader)
        accs.append(acc)
        bals.append(bal)
        if curves is not None:
            curves.append({
                "seed": seed,
                "fold": fold,
                "best_epoch": best_epoch,
                "epochs_run": len(val_bal),
                "val_bal_at_best": val_bal[best_epoch - 1],
                "val_bal": val_bal,
                "train_loss": train_loss,
                "test_acc": acc,
                "test_bal": bal,
            })
        print(
            f"    seed {seed} fold {fold}: acc={acc:.3f} bal={bal:.3f} "
            f"best_epoch={best_epoch} epochs_run={len(val_bal)}",
            flush=True,
        )
    return accs, bals


def score(model, loader):
    model.eval()
    pred, gold = [], []
    with torch.no_grad():
        for batch in loader:
            pred.extend(model(batch).argmax(dim=1).tolist())
            gold.extend(batch.y.tolist())
    return accuracy_score(gold, pred), balanced_accuracy_score(gold, pred)


def run_svm(x_or_text, y, seed, tfidf=False):
    set_seed(seed)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    accs, bals = [], []
    y = np.asarray(y)
    for fold, (tr_all, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr, va = val_split(tr_all, y, seed + fold)
        best_c, best_bal = 1.0, -1.0
        for c in (0.1, 1.0, 10.0):
            clf, xva, xte = fit_svm(x_or_text, y, tr, va, te, c, tfidf)
            pred = clf.predict(xva)
            bal = balanced_accuracy_score(y[va], pred)
            if bal > best_bal:
                best_bal, best_c = bal, c
                best = (clf, xte)
        pred = best[0].predict(best[1])
        accs.append(accuracy_score(y[te], pred))
        bals.append(balanced_accuracy_score(y[te], pred))
        print(f"    seed {seed} fold {fold}: acc={accs[-1]:.3f} bal={bals[-1]:.3f} C={best_c}", flush=True)
    return accs, bals


def fit_svm(x_or_text, y, tr, va, te, c, tfidf):
    clf = LinearSVC(C=c, class_weight="balanced", max_iter=5000, dual=False)
    if tfidf:
        vec = TfidfVectorizer(min_df=2, max_features=20000)
        xtr = vec.fit_transform([x_or_text[i] for i in tr])
        xva = vec.transform([x_or_text[i] for i in va])
        xte = vec.transform([x_or_text[i] for i in te])
    else:
        x = np.vstack(x_or_text)
        xtr, xva, xte = x[tr], x[va], x[te]
    clf.fit(xtr, y[tr])
    return clf, xva, xte


def summarise(accs, bals):
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_sd": float(np.std(accs, ddof=1)),
        "bal_mean": float(np.mean(bals)),
        "bal_sd": float(np.std(bals, ddof=1)),
        "n": len(accs),
        "bals": [float(b) for b in bals],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=[
        "e4mde_svm", "tfidf_svm", "gin_e4mde", "rgcn_e4mde", "rgcn_rich", "fusion",
    ])
    args = ap.parse_args()
    print("loading WordE4MDE", flush=True)
    kv = KeyedVectors.load(KV_PATH, mmap="r")
    rows, n_rel = load_models(kv)
    y = np.array([r["yi"] for r in rows])
    n_classes = int(y.max()) + 1
    texts = [r["text"] for r in rows]
    docs = [r["doc"] for r in rows]
    data_name = make_data(rows, rich=False)
    data_rich = make_data(rows, rich=True)
    results = {
        "n": len(rows),
        "n_categories": n_classes,
        "protocol": "5-fold x 2 seeds, model selection on validation only",
        "variants": {},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for name in args.variants:
        print(f"\n=== {name} ===", flush=True)
        accs, bals = [], []
        for seed in SEEDS:
            if name == "e4mde_svm":
                a, b = run_svm(docs, y, seed, tfidf=False)
            elif name == "tfidf_svm":
                a, b = run_svm(texts, y, seed, tfidf=True)
            elif name == "gin_e4mde":
                a, b = run_gnn(data_name, y, n_classes, n_rel, "gin", seed)
            elif name == "rgcn_e4mde":
                a, b = run_gnn(data_name, y, n_classes, n_rel, "rgcn", seed)
            elif name == "rgcn_rich":
                a, b = run_gnn(data_rich, y, n_classes, n_rel, "rgcn", seed)
            elif name == "fusion":
                a, b = run_gnn(data_rich, y, n_classes, n_rel, "fusion", seed)
            else:
                raise SystemExit("unknown " + name)
            accs += a
            bals += b
        results["variants"][name] = summarise(accs, bals)
        s = results["variants"][name]
        print(f"  SUMMARY {name}: acc {s['acc_mean']:.3f}+-{s['acc_sd']:.3f} "
              f"bal {s['bal_mean']:.3f}+-{s['bal_sd']:.3f}", flush=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
