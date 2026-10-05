"""Meta-model classification on ModelSet with MoEGL encodings (R1.4).

Every representation is produced by MoEGL from ONE Stage-1 cache; the
encodings differ only in their YAML configuration (configs/modelset/*.yaml):

  structure        homogeneous graph, one-hot node type only        (ms_structure.yaml)
  homo_names       homogeneous graph, node type + WordE4MDE(name)   (ms_names.yaml, NetworkX)
  hetero_names     heterogeneous graph, WordE4MDE(name) per type    (ms_names.yaml, HPyG)
  hetero_attrs     heterogeneous + structural attributes            (ms_names_attrs.yaml, HPyG)
  hetero_noedges   ablation: hetero_names with all edges removed

Baselines re-run on the SAME folds (best two methods of Lopez et al.,
MODELS 2022, and the WordE4MDE classifier of Lopez et al., MODELS 2023):
  tfidf_ffnn, tfidf_svm, w4mde_svm

Protocol: stratified 10-fold cross-validation (seed 0); inside each training
fold 10 % is held out for model selection (hyper-parameter grid for the
baselines, early stopping for the GNNs); metric = balanced accuracy on the
test fold. The same GNN architecture and training budget are used for every
MoEGL encoding.

Usage:
  python classify.py --prepared E:/Project/ModelSet/prepared --variant nodup \
      --methods structure homo_names hetero_names hetero_attrs hetero_noedges tfidf_ffnn tfidf_svm w4mde_svm
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)

from moegl import build_graphs, build_hetero, extract, load_config  # noqa: E402

CFG = os.path.join(PKG, "configs", "modelset")
GNN_METHODS = {"structure", "homo_names", "hetero_names", "hetero_attrs", "hetero_noedges"}


# --------------------------------------------------------------------------- data
def load_labels(prepared, variant):
    with open(os.path.join(prepared, f"labels_{variant}.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["name"] for r in rows]
    cats = sorted({r["category"] for r in rows})
    y = np.array([cats.index(r["category"]) for r in rows])
    return names, y, cats


def stage1(prepared, timings):
    cache_path = os.path.join(prepared, "moegl_cache.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    cfg = load_config(os.path.join(CFG, "ms_structure.yaml"))
    cfg.modelspath = os.path.join(prepared, "models")
    t0 = time.perf_counter()
    cache = extract(cfg)
    timings["stage1_s"] = time.perf_counter() - t0
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    return cache


def subset_cache(cache, names):
    keep = set(names)
    return {"models": [m for m in cache["models"] if m["name"] in keep], "meta": cache["meta"]}


def nx_to_data(g, type_index, feat_keys):
    """NetworkX graph from MoEGL -> homogeneous PyG Data (reverse edges added)."""
    from torch_geometric.data import Data

    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    rows = []
    for n in nodes:
        d = g.nodes[n]
        onehot = [0.0] * len(type_index)
        onehot[type_index[d["type"]]] = 1.0
        extra = []
        for k, width in feat_keys:
            v = d.get(k)
            extra.extend(v if isinstance(v, list) else [0.0] * width)
        rows.append(onehot + extra)
    src = [idx[u] for u, v in g.edges()]
    dst = [idx[v] for u, v in g.edges()]
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return Data(x=torch.tensor(rows, dtype=torch.float), edge_index=ei)


def build_homo(cache, names, cfg_file):
    cfg = load_config(os.path.join(CFG, cfg_file))
    cfg.output_format = "NetworkX"
    t0 = time.perf_counter()
    graphs = build_graphs(subset_cache(cache, names), cfg)
    t_enc = time.perf_counter() - t0
    types = sorted({d["type"] for g in graphs.values() for _, d in g.nodes(data=True)})
    type_index = {t: i for i, t in enumerate(types)}
    widths = {}
    for g in graphs.values():
        for _, d in g.nodes(data=True):
            for k, v in d.items():
                if k != "type" and isinstance(v, list):
                    widths.setdefault(k, len(v))
    feat_keys = sorted(widths.items())
    return [nx_to_data(graphs[n], type_index, feat_keys) for n in names], t_enc


def build_het(cache, names, cfg_file, drop_edges=False):
    import torch_geometric.transforms as T

    cfg = load_config(os.path.join(CFG, cfg_file))
    cfg.output_format = "HPyG"
    t0 = time.perf_counter()
    datas = build_hetero(subset_cache(cache, names), cfg)
    t_enc = time.perf_counter() - t0
    out = []
    to_undirected = T.ToUndirected()
    for n in names:
        d = datas[n]
        if drop_edges:
            for et in list(d.edge_types):
                del d[et]
        else:
            d = to_undirected(d)
        out.append(d)
    return out, t_enc


# --------------------------------------------------------------------------- GNNs
class HomoGNN(nn.Module):
    def __init__(self, in_dim, hidden, n_cls):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.inp = nn.Linear(in_dim, hidden)
        self.c1 = SAGEConv(hidden, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, n_cls))

    def forward(self, b):
        from torch_geometric.nn import global_max_pool, global_mean_pool

        h = F.relu(self.inp(b.x))
        h = F.relu(self.c1(h, b.edge_index)) + h
        h = F.relu(self.c2(h, b.edge_index)) + h
        g = torch.cat([global_mean_pool(h, b.batch), global_max_pool(h, b.batch)], dim=1)
        return self.out(g)


class HeteroGNN(nn.Module):
    """Type-specific input projections, relation-specific SAGE convolutions
    (HeteroConv), per-type mean/max read-out concatenated in a fixed order."""

    def __init__(self, in_dims, edge_types, hidden, n_cls):
        super().__init__()
        from torch_geometric.nn import HeteroConv, SAGEConv

        self.ntypes = sorted(in_dims)
        self.inp = nn.ModuleDict({t: nn.Linear(in_dims[t], hidden) for t in self.ntypes})
        self.edge_types = sorted(edge_types)
        self.convs = nn.ModuleList()
        if self.edge_types:
            for _ in range(2):
                self.convs.append(HeteroConv(
                    {et: SAGEConv((hidden, hidden), hidden) for et in self.edge_types}, aggr="sum"))
        self.out = nn.Sequential(nn.Linear(2 * hidden * len(self.ntypes), hidden), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(hidden, n_cls))
        self.hidden = hidden

    def forward(self, b):
        from torch_geometric.nn import global_max_pool, global_mean_pool

        h = {t: F.relu(self.inp[t](b[t].x)) for t in self.ntypes if t in b.node_types and b[t].num_nodes}
        for conv in self.convs:
            eidx = {et: b[et].edge_index for et in self.edge_types
                    if et in b.edge_types and et[0] in h and et[2] in h}
            new = conv(h, eidx) if eidx else {}
            h = {t: F.relu(new[t]) + h[t] if t in new else h[t] for t in h}
        n_graphs = b.num_graphs
        parts = []
        for t in self.ntypes:
            if t in h:
                bt = b[t].batch
                parts += [global_mean_pool(h[t], bt, size=n_graphs), global_max_pool(h[t], bt, size=n_graphs)]
            else:
                z = torch.zeros(n_graphs, self.hidden)
                parts += [z, z]
        return self.out(torch.cat(parts, dim=1))


def align_hetero(datas):
    """Give every graph every node type (empty tensors) so batching is uniform."""
    dims, etypes = {}, set()
    for d in datas:
        for t in d.node_types:
            dims.setdefault(t, d[t].x.size(1))
        etypes.update(d.edge_types)
    for d in datas:
        for t, w in dims.items():
            if t not in d.node_types:
                d[t].x = torch.zeros(0, w)
                d[t].node_id = torch.zeros(0, dtype=torch.long)
                d[t].orig_id = torch.zeros(0, dtype=torch.long)
        for et in etypes:
            if et not in d.edge_types:
                d[et].edge_index = torch.zeros(2, 0, dtype=torch.long)
    return dims, etypes


def train_gnn(method, datas, y, tr, va, te, n_cls, seed, epochs, hidden=128):
    from torch_geometric.loader import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    for i, d in enumerate(datas):
        d.y = torch.tensor([int(y[i])])
    if method in ("structure", "homo_names"):
        model = HomoGNN(datas[0].x.size(1), hidden, n_cls)
    else:
        dims, etypes = align_hetero(datas)
        model = HeteroGNN(dims, etypes, hidden, n_cls)
    counts = np.bincount(y[tr], minlength=n_cls).astype(float)
    weight = torch.tensor(np.where(counts > 0, counts.sum() / (n_cls * np.maximum(counts, 1)), 0.0),
                          dtype=torch.float)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader([datas[i] for i in tr], batch_size=32, shuffle=True, generator=g)

    def predict(idx):
        model.eval()
        preds = []
        with torch.no_grad():
            for b in DataLoader([datas[i] for i in idx], batch_size=128):
                preds.append(model(b).argmax(1))
        return torch.cat(preds).numpy()

    best, best_state, patience = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for b in tr_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(b), b.y, weight=weight)
            loss.backward()
            opt.step()
        score = balanced_accuracy_score(y[va], predict(va))
        if score > best:
            best, patience = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
                break
    model.load_state_dict(best_state)
    return balanced_accuracy_score(y[te], predict(te)), {"val": best, "epochs": ep + 1}


# --------------------------------------------------------------------------- baselines
def tfidf_baseline(kind, docs, y, tr, va, te, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    vec = TfidfVectorizer(analyzer=lambda d: d)
    x_tr = vec.fit_transform([docs[i] for i in tr])
    x_va = vec.transform([docs[i] for i in va])
    x_te = vec.transform([docs[i] for i in te])
    if kind == "ffnn":
        grid = [{"hidden_layer_sizes": (h,)} for h in (50, 100, 150, 200)]
        make = lambda p: MLPClassifier(max_iter=300, random_state=seed, **p)  # noqa: E731
    else:
        grid = [{"C": c, "kernel": k} for c in (0.01, 0.1, 1, 10, 100) for k in ("linear", "rbf")]
        make = lambda p: SVC(**p)  # noqa: E731
    best = max(grid, key=lambda p: balanced_accuracy_score(y[va], make(p).fit(x_tr, y[tr]).predict(x_va)))
    clf = make(best).fit(x_tr, y[tr])
    return balanced_accuracy_score(y[te], clf.predict(x_te)), {"best": str(best)}


def w4mde_svm(docs, y, tr, va, te, kv):
    from sklearn.svm import SVC

    def emb(d):
        v = [kv[t] for t in d if t in kv.key_to_index]
        return np.mean(v, axis=0) if v else np.zeros(kv.vector_size)

    x = np.stack([emb(d) for d in docs])
    grid = [{"C": c, "kernel": k} for c in (0.01, 0.1, 1, 10, 100) for k in ("linear", "rbf")]
    best = max(grid, key=lambda p: balanced_accuracy_score(y[va], SVC(**p).fit(x[tr], y[tr]).predict(x[va])))
    clf = SVC(**best).fit(x[tr], y[tr])
    return balanced_accuracy_score(y[te], clf.predict(x[te])), {"best": str(best)}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--variant", default="nodup", choices=["nodup", "all"])
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--only-folds", type=int, default=0, help="debug: run only the first k folds")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    names, y, cats = load_labels(args.prepared, args.variant)
    n_cls = len(cats)
    timings = {}
    cache = stage1(args.prepared, timings)
    by_name = {m["name"]: m for m in cache["models"]}
    missing = [n for n in names if n not in by_name]
    if missing:  # models that failed to load are reported and excluded
        keep = [i for i, n in enumerate(names) if n in by_name]
        names, y = [names[i] for i in keep], y[keep]
    with open(os.path.join(args.prepared, "tokens.json"), encoding="utf-8") as f:
        tokens = json.load(f)
    docs = [tokens[n] for n in names]
    kv = None

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = []
    for tr_all, te in skf.split(names, y):
        tr, va = train_test_split(tr_all, test_size=0.1, stratify=y[tr_all], random_state=args.seed)
        folds.append((tr, va, te))
    if args.only_folds:
        folds = folds[: args.only_folds]

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"modelset_{args.variant}.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    results["dataset"] = {"variant": args.variant, "models": len(names), "categories": n_cls,
                          "failed_to_load": missing, "folds": args.folds, "seed": args.seed}
    results.setdefault("timings", {}).update(timings)

    for method in args.methods:
        t0 = time.perf_counter()
        datas, t_enc = None, None
        if method == "structure":
            datas, t_enc = build_homo(cache, names, "ms_structure.yaml")
        elif method == "homo_names":
            datas, t_enc = build_homo(cache, names, "ms_names.yaml")
        elif method == "hetero_names":
            datas, t_enc = build_het(cache, names, "ms_names.yaml")
        elif method == "hetero_attrs":
            datas, t_enc = build_het(cache, names, "ms_names_attrs.yaml")
        elif method == "hetero_noedges":
            datas, t_enc = build_het(cache, names, "ms_names.yaml", drop_edges=True)
        elif method == "w4mde_svm" and kv is None:
            from worde4mde import load_embeddings
            kv = load_embeddings("sgram-mde")

        scores, info = [], []
        for k, (tr, va, te) in enumerate(folds):
            if method in GNN_METHODS:
                s, inf = train_gnn(method, datas, y, tr, va, te, n_cls, args.seed + k, args.epochs)
            elif method == "tfidf_ffnn":
                s, inf = tfidf_baseline("ffnn", docs, y, tr, va, te, args.seed)
            elif method == "tfidf_svm":
                s, inf = tfidf_baseline("svm", docs, y, tr, va, te, args.seed)
            elif method == "w4mde_svm":
                s, inf = w4mde_svm(docs, y, tr, va, te, kv)
            else:
                raise ValueError(method)
            scores.append(s)
            info.append(inf)
            print(f"{method} fold {k}: {s:.4f} {inf}", flush=True)
        results[method] = {
            "balanced_accuracy": scores, "mean": float(np.mean(scores)), "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "info": info, "moegl_stage2_s": t_enc, "wall_s": time.perf_counter() - t0,
        }
        print(f"== {method}: {results[method]['mean']:.4f} +- {results[method]['sd']:.4f}", flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
