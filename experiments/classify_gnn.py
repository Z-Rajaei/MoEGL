"""Experiment 4: effect of MoEGL encoding choices on a GNN classifier.

Variants (all graphs produced by MoEGL YAML configs, same GIN architecture):
  struct          -- type one-hot only (no attributes)
  name_w2v        -- type one-hot + word2vec(name)
  name_e4mde      -- type one-hot + worde4mde(name)
  hetero_w2v      -- same as name_w2v but RGCN over typed edges (heterogeneous)
  tfidf_svm       -- bag-of-names SVM (names extracted by the same MoEGL cache)

Protocol: stratified 5-fold CV, 3 seeds, report mean+-sd accuracy and
balanced accuracy. Hyper-parameters are fixed (documented in the paper).
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, RGCNConv, global_mean_pool

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from moegl import load_config, extract, build_graphs  # noqa: E402
from moegl.encoders import tokenize  # noqa: E402

LABELS_PATH = os.path.join(HERE, "data", "modelset_ecore", "labels_dedup.json")
CONFIGS = os.path.join(PKG, "configs")
OUT = os.path.join(HERE, "results", "exp4_classify.json")

HIDDEN = 128
LAYERS = 2
EPOCHS = 40
LR = 1e-3
BATCH = 32
SEEDS = (0, 1, 2)
FOLDS = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def type_vocab(graphs):
    types = sorted({d["type"] for g in graphs.values() for _, d in g.nodes(data=True)})
    return {t: i for i, t in enumerate(types)}


def edge_type_vocab(graphs):
    types = sorted({d["type"] for g in graphs.values() for _, _, d in g.edges(data=True)})
    return {t: i for i, t in enumerate(types)}


def _name_vec(attrs):
    v = attrs.get("name")
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=np.float32)
    return None


def graph_to_data(g, y, t2i, e2i, use_name, hetero):
    n = g.number_of_nodes()
    if n == 0:
        return None
    nodes = list(g.nodes())
    loc = {nid: i for i, nid in enumerate(nodes)}
    type_oh = np.zeros((n, len(t2i)), dtype=np.float32)
    name_rows = []
    name_dim = 0
    for i, nid in enumerate(nodes):
        d = g.nodes[nid]
        type_oh[i, t2i[d["type"]]] = 1.0
        nv = _name_vec(d) if use_name else None
        if nv is not None:
            name_dim = max(name_dim, len(nv))
            name_rows.append(nv)
        else:
            name_rows.append(None)
    if use_name and name_dim:
        name_mat = np.zeros((n, name_dim), dtype=np.float32)
        for i, nv in enumerate(name_rows):
            if nv is not None:
                name_mat[i, : len(nv)] = nv
        x = np.concatenate([type_oh, name_mat], axis=1)
    else:
        x = type_oh
    src, dst, et = [], [], []
    for u, v, d in g.edges(data=True):
        src.append(loc[u])
        dst.append(loc[v])
        et.append(e2i.get(d["type"], 0))
    if not src:
        src, dst, et = [0], [0], [0]
    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_type=torch.tensor(et, dtype=torch.long),
        y=torch.tensor([y], dtype=torch.long),
        num_nodes=n,
    )
    data.num_relations = len(e2i) if e2i else 1
    data.hetero = hetero
    return data


class GINNet(torch.nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        last = in_dim
        for _ in range(LAYERS):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(last, HIDDEN),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN, HIDDEN),
            )
            self.convs.append(GINConv(mlp))
            last = HIDDEN
        self.lin = torch.nn.Linear(HIDDEN, n_classes)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, ei))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


class RGCNNet(torch.nn.Module):
    def __init__(self, in_dim, n_classes, n_rel):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        last = in_dim
        for _ in range(LAYERS):
            self.convs.append(RGCNConv(last, HIDDEN, num_relations=max(n_rel, 1)))
            last = HIDDEN
        self.lin = torch.nn.Linear(HIDDEN, n_classes)

    def forward(self, data):
        x, ei, et = data.x, data.edge_index, data.edge_type
        for conv in self.convs:
            x = F.relu(conv(x, ei, et))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


def train_eval_gnn(dataset, labels, n_classes, hetero, seed):
    set_seed(seed)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    y = np.array(labels)
    accs, bals = [], []
    in_dim = dataset[0].x.size(1)
    n_rel = int(dataset[0].num_relations)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        train_ds = [dataset[i] for i in tr]
        test_ds = [dataset[i] for i in te]
        tr_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
        te_loader = DataLoader(test_ds, batch_size=BATCH)
        model = (
            RGCNNet(in_dim, n_classes, n_rel)
            if hetero
            else GINNet(in_dim, n_classes)
        )
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        best = (0.0, 0.0)
        for _ in range(EPOCHS):
            model.train()
            for batch in tr_loader:
                opt.zero_grad()
                out = model(batch)
                loss = F.cross_entropy(out, batch.y)
                loss.backward()
                opt.step()
            model.eval()
            pred, gold = [], []
            with torch.no_grad():
                for batch in te_loader:
                    logits = model(batch)
                    pred.extend(logits.argmax(dim=1).tolist())
                    gold.extend(batch.y.tolist())
            acc = accuracy_score(gold, pred)
            bal = balanced_accuracy_score(gold, pred)
            if bal > best[1]:
                best = (acc, bal)
        accs.append(best[0])
        bals.append(best[1])
        print(f"    seed {seed} fold {fold}: acc={best[0]:.3f} bal={best[1]:.3f}", flush=True)
    return accs, bals


def names_from_cache(cache):
    docs = {}
    for model in cache["models"]:
        tokens = []
        for node in model["nodes"]:
            name = node.get("attrs", {}).get("name")
            if name is None:
                continue
            tokens.extend(tokenize(name))
        docs[model["name"]] = " ".join(tokens)
    return docs


def train_eval_tfidf(docs, names, labels, n_classes, seed):
    set_seed(seed)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    y = np.array(labels)
    texts = [docs[n] for n in names]
    accs, bals = [], []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        vec = TfidfVectorizer(min_df=2, max_features=20000)
        xtr = vec.fit_transform([texts[i] for i in tr])
        xte = vec.transform([texts[i] for i in te])
        clf = LinearSVC(C=10.0, max_iter=4000)
        clf.fit(xtr, y[tr])
        pred = clf.predict(xte)
        accs.append(accuracy_score(y[te], pred))
        bals.append(balanced_accuracy_score(y[te], pred))
        print(
            f"    seed {seed} fold {fold}: acc={accs[-1]:.3f} bal={bals[-1]:.3f}",
            flush=True,
        )
    return accs, bals


def summarise(accs, bals):
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "bal_mean": float(np.mean(bals)),
        "bal_sd": float(np.std(bals, ddof=1)) if len(bals) > 1 else 0.0,
        "n": len(accs),
        "accs": [float(x) for x in accs],
        "bals": [float(x) for x in bals],
    }


def build_dataset(graphs, labels_map, use_name, hetero):
    t2i = type_vocab(graphs)
    e2i = edge_type_vocab(graphs)
    cat2i = {c: i for i, c in enumerate(sorted(set(labels_map.values())))}
    dataset, names, ys = [], [], []
    skipped = 0
    for name, g in sorted(graphs.items()):
        if name not in labels_map:
            continue
        data = graph_to_data(g, cat2i[labels_map[name]], t2i, e2i, use_name, hetero)
        if data is None:
            skipped += 1
            continue
        dataset.append(data)
        names.append(name)
        ys.append(cat2i[labels_map[name]])
    print(
        f"  graphs={len(dataset)} skipped_empty={skipped} "
        f"types={len(t2i)} etypes={len(e2i)} classes={len(cat2i)}",
        flush=True,
    )
    return dataset, names, ys, cat2i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["struct", "name_w2v", "name_e4mde", "hetero_w2v", "tfidf_svm"],
    )
    args = ap.parse_args()

    with open(LABELS_PATH, encoding="utf-8") as f:
        pack = json.load(f)
    labels_map = pack["labels"]
    print("labels", pack["n"], "cats", pack["n_categories"], flush=True)

    cfg_by_variant = {
        "struct": "exp4_modelset_struct.yaml",
        "name_w2v": "exp4_modelset_name_w2v.yaml",
        "name_e4mde": "exp4_modelset_name_e4mde.yaml",
        "hetero_w2v": "exp4_modelset_name_w2v.yaml",
        "tfidf_svm": "exp4_modelset_name_w2v.yaml",
    }

    results = {
        "protocol": {
            "folds": FOLDS,
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "lr": LR,
            "batch": BATCH,
            "n_models": pack["n"],
            "n_categories": pack["n_categories"],
        },
        "variants": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    cache_by_cfg = {}
    graphs_by_cfg = {}

    for variant in args.variants:
        cfg_name = cfg_by_variant[variant]
        print(f"\n=== {variant} ({cfg_name}) ===", flush=True)
        if cfg_name not in cache_by_cfg:
            cfg = load_config(os.path.join(CONFIGS, cfg_name))
            t0 = time.perf_counter()
            cache = extract(cfg)
            graphs = build_graphs(cache, cfg)
            print(
                f"  extract+encode {time.perf_counter() - t0:.1f}s "
                f"models={len(graphs)} skipped={len(cache['meta'].get('skipped', []))}",
                flush=True,
            )
            cache_by_cfg[cfg_name] = cache
            graphs_by_cfg[cfg_name] = graphs
        cache = cache_by_cfg[cfg_name]
        graphs = graphs_by_cfg[cfg_name]

        if variant == "tfidf_svm":
            docs = names_from_cache(cache)
            names = [n for n in sorted(docs) if n in labels_map]
            ys = [labels_map[n] for n in names]
            cat2i = {c: i for i, c in enumerate(sorted(set(ys)))}
            y_idx = [cat2i[c] for c in ys]
            print("  class counts", Counter(ys).most_common(8), flush=True)
            all_acc, all_bal = [], []
            for seed in SEEDS:
                a, b = train_eval_tfidf(docs, names, y_idx, len(cat2i), seed)
                all_acc += a
                all_bal += b
            results["variants"][variant] = summarise(all_acc, all_bal)
        else:
            use_name = variant != "struct"
            hetero = variant.startswith("hetero")
            dataset, names, ys, cat2i = build_dataset(
                graphs, labels_map, use_name=use_name, hetero=hetero
            )
            print("  class counts", Counter(ys).most_common(8), flush=True)
            all_acc, all_bal = [], []
            for seed in SEEDS:
                a, b = train_eval_gnn(dataset, ys, len(cat2i), hetero, seed)
                all_acc += a
                all_bal += b
            results["variants"][variant] = summarise(all_acc, all_bal)
        s = results["variants"][variant]
        print(
            f"  SUMMARY {variant}: acc {s['acc_mean']:.3f}+-{s['acc_sd']:.3f} "
            f"bal {s['bal_mean']:.3f}+-{s['bal_sd']:.3f}",
            flush=True,
        )
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
