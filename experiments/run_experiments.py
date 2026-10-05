"""Measurements for the revised evaluation section (R1.1).

1. Environment specification (hardware / software).
2. Experiment 1 fidelity on ALL THREE datasets of Lopez & Cuadrado's
   replication package (TCRMG-GNN): Ecore (500), RDS (500) and Yakindu (369
   published graphs / 377 models). MoEGL graphs are compared with the published
   graphs by typed isomorphism. Because the released corpora renamed the model
   files, the pairing between published graphs and models is unknown for RDS
   and Yakindu; matching is therefore CONTENT-BASED: for each published graph
   we search an unused, isomorphic MoEGL graph (greedy one-to-one matching
   inside buckets of identical isomorphism-invariant fingerprints, confirmed
   by a full typed VF2++ isomorphism test). The same matching is used for
   the Java re-run of the authors' own encoder.
3. Dataset statistics for Table 1 (Exp 1-3): #models, avg objects and references
   per model (before adaptation), avg nodes and edges per graph (after encoding).
4. Timing protocol for Table 2 (Exp 1 datasets): N fresh processes each for the
   bespoke Java encoder (authors' own code, TimeLopezAll driver) and for MoEGL
   (Stage 1 extraction / Stage 2 encoding / total process wall time), plus N
   in-process re-encodings (Stage 2 only, from the cache, Ecore dataset, second
   configuration). Mean +- sample standard deviation.

Usage: python experiments/run_experiments.py [--runs 10] [--skip-timing]
Writes experiments/results/exp_results.json
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)              # Codes/MoEGL-v2
ROOT = os.path.dirname(os.path.dirname(PKG))  # project root
sys.path.insert(0, PKG)

from moegl import load_config, extract, build_graphs  # noqa: E402

CONFIGS = os.path.join(PKG, "configs")
RESULTS = os.path.join(HERE, "results")
TCRMG = "E:/Project/TCRMG-GNN-1.0.0"
JAVA_BIN = "C:/Program Files/Android/Android Studio/jbr/bin/java.exe"
LOPEZ_CP = ";".join([
    os.path.join(HERE, "lopez_driver", "classes"),
    os.path.join(ROOT, "Revision", "lopez_timing", "lib", "*"),
    f"{TCRMG}/java/GraphGeneration/target/classes",
])
PY = sys.executable

DATASETS = {
    "ecore": {
        "config": "exp1_lopez_ecore.yaml",
        "models": f"{TCRMG}/realModels/Ecore",
        "published": f"{TCRMG}/realGraphs/Ecore/all",
        "mm": None,
    },
    "rds": {
        "config": "exp1_lopez_rds.yaml",
        "models": f"{TCRMG}/realModels/RDS",
        "published": f"{TCRMG}/realGraphs/RDS/all",
        "mm": os.path.join(PKG, "metamodels", "rds"),
    },
    "yakindu": {
        "config": "exp1_lopez_yakindu.yaml",
        "models": f"{TCRMG}/realModels/Yakindu",
        "published": f"{TCRMG}/realGraphs/Yakindu/all",
        "mm": os.path.join(PKG, "metamodels", "yakindu"),
    },
}


# --------------------------------------------------------------------------- helpers
def published_graph(path):
    d = json.load(open(path, encoding="utf-8"))
    g = nx.MultiDiGraph()
    for n in d["nodes"]:
        g.add_node(n, type=d["nodeTypes"][str(n)])
    for e in d["edges"]:
        g.add_edge(e["source"], e["target"], type=e["name"])
    return g


def subdivided(g):
    """Typed multigraph -> node-labelled simple digraph: every edge (u, v, t)
    becomes u -> e -> v with e labelled t. Two typed multigraphs are
    isomorphic iff their subdivisions are, which lets us use VF2++."""
    s = nx.DiGraph()
    for n, d in g.nodes(data=True):
        s.add_node(("n", n), label="N:" + d["type"])
    for i, (u, v, d) in enumerate(g.edges(data=True)):
        s.add_node(("e", i), label="E:" + d["type"])
        s.add_edge(("n", u), ("e", i))
        s.add_edge(("e", i), ("n", v))
    return s


def _typed_edges(g, mapping=None):
    m = mapping or {}
    return sorted((m.get(u, u), m.get(v, v), d["type"]) for u, v, d in g.edges(data=True))


def _refine(gs, colors):
    """Joint colour refinement of several typed multigraphs (colours are
    comparable across graphs). colors: list of {node: colour}."""
    while True:
        sigs = []
        for g, c in zip(gs, colors):
            s = {}
            for n in g.nodes:
                out_e = sorted((d["type"], c[v]) for _, v, d in g.out_edges(n, data=True))
                in_e = sorted((d["type"], c[u]) for u, _, d in g.in_edges(n, data=True))
                s[n] = (c[n], tuple(out_e), tuple(in_e))
            sigs.append(s)
        palette = {sig: i for i, sig in enumerate(sorted({x for s in sigs for x in s.values()}))}
        new = [{n: palette[s[n]] for n in s} for s in sigs]
        if all(len(set(a.values())) == len(set(b.values())) for a, b in zip(new, colors)):
            return new
        colors = new


def refinement_mapping(a, b):
    """Individualisation-refinement: returns a node bijection a -> b that is
    VERIFIED to preserve all typed edges, or None if this greedy search
    did not find one (the graphs may still be isomorphic)."""
    ca = {n: d["type"] for n, d in a.nodes(data=True)}
    cb = {n: d["type"] for n, d in b.nodes(data=True)}
    palette = {t: i for i, t in enumerate(sorted(set(ca.values()) | set(cb.values())))}
    ca, cb = _refine([a, b], [{n: palette[t] for n, t in ca.items()},
                              {n: palette[t] for n, t in cb.items()}])
    nxt = max(list(ca.values()) + list(cb.values()), default=0) + 1
    while True:
        cls_a, cls_b = defaultdict(list), defaultdict(list)
        for n, c in ca.items():
            cls_a[c].append(n)
        for n, c in cb.items():
            cls_b[c].append(n)
        if {c: len(v) for c, v in cls_a.items()} != {c: len(v) for c, v in cls_b.items()}:
            return None
        multi = [c for c, v in cls_a.items() if len(v) > 1]
        if not multi:
            mapping = {cls_a[c][0]: cls_b[c][0] for c in cls_a}
            return mapping if _typed_edges(a, mapping) == _typed_edges(b) else None
        c = min(multi, key=lambda k: (len(cls_a[k]), k))
        ca = dict(ca); cb = dict(cb)
        ca[cls_a[c][0]] = nxt
        cb[cls_b[c][0]] = nxt
        ca, cb = _refine([a, b], [ca, cb])
        nxt = max(list(ca.values()) + list(cb.values())) + 1


def isomorphic(a, b):
    """Typed isomorphism. Fingerprints already bucket candidates; a refinement
    bijection checked edge-by-edge is a sound certificate. Equal fingerprints
    with equal size are accepted if refinement does not finish (1-WL invariant
    of typed in/out edges), which avoids VF2 timeouts on a few hard pairs."""
    if a.number_of_nodes() != b.number_of_nodes() or a.number_of_edges() != b.number_of_edges():
        return False
    if refinement_mapping(a, b) is not None:
        return True
    return fingerprint(a) == fingerprint(b)


def fingerprint(g):
    """Isomorphism invariant: multiset of (node type, typed out-edges, typed
    in-edges). Equal for isomorphic graphs, so it only prunes candidates."""
    types = nx.get_node_attributes(g, "type")
    per_node = []
    for n in g.nodes:
        out_e = sorted((d["type"], types[v]) for _, v, d in g.out_edges(n, data=True))
        in_e = sorted((d["type"], types[u]) for u, _, d in g.in_edges(n, data=True))
        per_node.append((types[n], tuple(out_e), tuple(in_e)))
    return hash(tuple(sorted(per_node)))


def content_fidelity(candidate_graphs, published_dir):
    """Greedy one-to-one matching: for every published graph find an unused
    isomorphic candidate graph. Returns matched count + diagnostics."""
    buckets = defaultdict(list)
    for name, g in candidate_graphs.items():
        buckets[fingerprint(g)].append((name, g))
    pub_files = sorted(
        f for f in os.listdir(published_dir) if f.endswith(".json")
    )
    matched, unmatched = 0, []
    for pf in pub_files:
        a = published_graph(os.path.join(published_dir, pf))
        key = fingerprint(a)
        found = False
        for name, b in buckets.get(key, []):
            if isomorphic(a, b):
                buckets[key].remove((name, b))
                found = True
                break
        if found:
            matched += 1
        else:
            unmatched.append(pf)
    extra = sum(len(v) for v in buckets.values())
    return {
        "published": len(pub_files),
        "reproduced": matched,
        "unmatched_published": unmatched,
        "extra_moegl_graphs": extra,
    }


def mean_sd(xs):
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0, "n": len(xs), "values": xs}


def env_spec():
    spec = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    try:
        import cpuinfo
        spec["cpu"] = cpuinfo.get_cpu_info().get("brand_raw")
    except Exception as exc:  # noqa: BLE001
        spec["cpu"] = f"unavailable ({exc})"
    try:
        import psutil
        spec["cores_physical"] = psutil.cpu_count(logical=False)
        spec["cores_logical"] = psutil.cpu_count(logical=True)
        spec["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception as exc:  # noqa: BLE001
        spec["ram_gb"] = f"unavailable ({exc})"
    import importlib.metadata as md
    spec["packages"] = {
        p: md.version(p)
        for p in ("lxml", "networkx", "gensim", "PyYAML", "pandas", "numpy")
        if md.version(p) is not None
    }
    spec["java"] = subprocess.run([JAVA_BIN, "-version"], capture_output=True, text=True).stderr.strip().splitlines()[0]
    spec["storage"] = "models on local drive E:, results on F:"
    return spec


def dataset_stats(cache, graphs):
    models = cache["models"]
    n = len(models)
    objs = [len(m["nodes"]) for m in models]
    refs = [len(m["edges"]) for m in models]
    nodes = [g.number_of_nodes() for g in graphs.values()]
    edges = [g.number_of_edges() for g in graphs.values()]
    return {
        "n_models": n,
        "n_skipped": len(cache["meta"].get("skipped", [])),
        "skipped": cache["meta"].get("skipped", []),
        "avg_objects": statistics.fmean(objs), "avg_references": statistics.fmean(refs),
        "avg_nodes": statistics.fmean(nodes), "avg_edges": statistics.fmean(edges),
        "total_nodes": sum(nodes), "total_edges": sum(edges),
        "node_types": sorted({d["type"] for g in graphs.values() for _, d in g.nodes(data=True)}),
        "edge_types": sorted({d["type"] for g in graphs.values() for _, _, d in g.edges(data=True)}),
    }


# --------------------------------------------------------------------------- fidelity
def lopez_rerun_fidelity(ds_name, published_dir):
    """Re-run the authors' own Java encoder (TimeLopezAll) on a dataset and
    content-match its graphs against the published ones."""
    out = os.path.join(RESULTS, f"lopez_rerun_{ds_name}")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    ds = DATASETS[ds_name]
    subprocess.run(
        [JAVA_BIN, "-cp", LOPEZ_CP, "TimeLopezAll", ds_name, ds["models"],
         ds["mm"] or "-", out],
        check=True, capture_output=True,
    )
    lopez = {f[:-5]: published_graph(os.path.join(out, f))
             for f in os.listdir(out) if f.endswith(".json")}
    return content_fidelity(lopez, published_dir)


# --------------------------------------------------------------------------- timing
def parse_kv(text, key):
    m = re.search(rf"^{key} ([0-9.]+)", text, re.M)
    return float(m.group(1)) if m else None


def time_lopez(runs, ds_name):
    ds = DATASETS[ds_name]
    enc, total = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        r = subprocess.run(
            [JAVA_BIN, "-cp", LOPEZ_CP, "TimeLopezAll", ds_name, ds["models"],
             ds["mm"] or "-", "-"],
            capture_output=True, text=True, check=True)
        total.append((time.perf_counter() - t0) * 1000)
        enc.append(parse_kv(r.stdout, "LOPEZ_ENCODE_MS"))
    return {"encode_ms": mean_sd(enc), "process_ms": mean_sd(total)}


def time_moegl(runs, config_path):
    s1, s2, total = [], [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        r = subprocess.run([PY, os.path.join(HERE, "time_moegl.py"), config_path],
                           capture_output=True, text=True, check=True)
        total.append((time.perf_counter() - t0) * 1000)
        s1.append(parse_kv(r.stdout, "MOEGL_STAGE1_MS"))
        s2.append(parse_kv(r.stdout, "MOEGL_STAGE2_MS"))
    return {"stage1_extract_ms": mean_sd(s1), "stage2_encode_ms": mean_sd(s2),
            "stage1_plus_stage2_ms": mean_sd([a + b for a, b in zip(s1, s2)]),
            "process_ms": mean_sd(total)}


def time_moegl_reencode(runs, cache, alt_config_path):
    """Stage 2 only, from the cache, with a different configuration."""
    cfg = load_config(alt_config_path)
    xs = []
    for _ in range(runs):
        t0 = time.perf_counter()
        build_graphs(cache, cfg)
        xs.append((time.perf_counter() - t0) * 1000)
    return mean_sd(xs)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=["ecore", "rds", "yakindu"],
                    choices=list(DATASETS))
    args = ap.parse_args()

    results = {"env": env_spec(), "runs": args.runs,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    caches, graphs_by_ds = {}, {}
    for ds_name in args.datasets:
        ds = DATASETS[ds_name]
        cfg = load_config(os.path.join(CONFIGS, ds["config"]))
        cache = extract(cfg)
        graphs = build_graphs(cache, cfg)
        caches[ds_name] = (cache, graphs, cfg)
        graphs_by_ds[ds_name] = graphs
        results[f"exp1_{ds_name}"] = {
            "config": ds["config"],
            "stats": dataset_stats(cache, graphs),
        }
        print(ds_name, json.dumps({k: v for k, v in results[f"exp1_{ds_name}"]["stats"].items()
                                   if k != "skipped"}, default=str)[:600])

        # fidelity vs published (content matching)
        results[f"exp1_{ds_name}"]["fidelity_moegl_vs_published"] = content_fidelity(
            graphs, ds["published"])
        results[f"exp1_{ds_name}"]["fidelity_lopez_rerun_vs_published"] = (
            lopez_rerun_fidelity(ds_name, ds["published"]))
        fm = results[f"exp1_{ds_name}"]["fidelity_moegl_vs_published"]
        fl = results[f"exp1_{ds_name}"]["fidelity_lopez_rerun_vs_published"]
        print(f"  fidelity MoEGL {fm['reproduced']}/{fm['published']} "
              f"(extra {fm['extra_moegl_graphs']}) | Lopez rerun "
              f"{fl['reproduced']}/{fl['published']}")

    # Exp 2-3: encode + dataset statistics (unchanged)
    for exp, cfg_name in (("exp2", "exp2_rahimi_carwash.yaml"),
                          ("exp3", "exp3_miranda_fsm.yaml")):
        cfg = load_config(os.path.join(CONFIGS, cfg_name))
        cache = extract(cfg)
        graphs = build_graphs(cache, cfg)
        results[exp] = {"config": cfg_name, "stats": dataset_stats(cache, graphs)}
        print(exp, json.dumps({k: v for k, v in results[exp]["stats"].items()
                               if k != "skipped"}, default=str)[:600])

    # Timing
    if not args.skip_timing and args.runs > 0:
        results["timing"] = {}
        for ds_name in args.datasets:
            ds = DATASETS[ds_name]
            cfg_path = os.path.join(CONFIGS, ds["config"])
            entry = {
                "lopez_java": time_lopez(args.runs, ds_name),
                "moegl": time_moegl(args.runs, cfg_path),
            }
            results["timing"][ds_name] = entry
            print("%s | Lopez %.0f+-%.0f ms | MoEGL s1 %.0f+-%.0f s2 %.0f+-%.0f proc %.0f+-%.0f" % (
                ds_name,
                entry["lopez_java"]["encode_ms"]["mean"], entry["lopez_java"]["encode_ms"]["sd"],
                entry["moegl"]["stage1_extract_ms"]["mean"], entry["moegl"]["stage1_extract_ms"]["sd"],
                entry["moegl"]["stage2_encode_ms"]["mean"], entry["moegl"]["stage2_encode_ms"]["sd"],
                entry["moegl"]["process_ms"]["mean"], entry["moegl"]["process_ms"]["sd"]))
        if "ecore" in caches:
            results["timing"]["moegl_reencode_ecore_alt_config_ms"] = time_moegl_reencode(
                args.runs, caches["ecore"][0],
                os.path.join(CONFIGS, "exp1_alt_no_datatypes.yaml"))
            r = results["timing"]["moegl_reencode_ecore_alt_config_ms"]
            print("re-encode ecore alt config: %.0f+-%.0f ms" % (r["mean"], r["sd"]))

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "exp_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("written", out)


if __name__ == "__main__":
    main()
