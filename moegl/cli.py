"""Command line interface: python -m moegl config.yaml [--cache DIR] [--out DIR]."""

import argparse
import json
import os
import time

import networkx as nx
from networkx.readwrite import json_graph

from .config import load_config
from .extract import extract, load_cache, save_cache
from .graphs import build_graphs


def _write_graphs(graphs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, g in graphs.items():
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_graph.node_link_data(g, edges="edges"), f)


def _write_hetero(datas, out_dir):
    import torch

    os.makedirs(out_dir, exist_ok=True)
    for name, d in datas.items():
        torch.save(d, os.path.join(out_dir, f"{name}.pt"))


def _summary(cache, n_written, t_extract, t_graphs):
    models = cache.get("models", [])
    n_nodes = sum(len(m["nodes"]) for m in models)
    n_edges = sum(len(m["edges"]) for m in models)
    n = len(models) or 1
    skipped = len((cache.get("meta") or {}).get("skipped", []))
    print(
        f"models={len(models)} skipped={skipped} written={n_written} "
        f"avg_nodes={n_nodes / n:.1f} avg_edges={n_edges / n:.1f} "
        f"stage1_extract={t_extract:.2f}s stage2_graphs={t_graphs:.2f}s"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="moegl")
    ap.add_argument("config", help="path to the MoEGL YAML configuration")
    ap.add_argument("--cache", default="./cache", help="cache directory")
    ap.add_argument("--out", default="./out", help="output directory")
    ap.add_argument(
        "--stage",
        choices=["extract", "graphs", "all"],
        default="all",
        help="which stage(s) to run",
    )
    args = ap.parse_args(argv)

    config = load_config(args.config)
    t_extract = t_graphs = 0.0
    cache = None

    if args.stage in ("extract", "all"):
        t0 = time.perf_counter()
        cache = extract(config)
        save_cache(cache, args.cache)
        t_extract = time.perf_counter() - t0

    n_written = 0
    if args.stage in ("graphs", "all"):
        if cache is None:
            cache = load_cache(args.cache)
        t0 = time.perf_counter()
        if config.output_format == "HPyG":
            from .hetero import build_hetero

            datas = build_hetero(cache, config)
            _write_hetero(datas, args.out)
        else:
            graphs = build_graphs(cache, config)
            _write_graphs(graphs, args.out)
        t_graphs = time.perf_counter() - t0
        n_written = len(cache["models"])

    _summary(cache, n_written, t_extract, t_graphs)


if __name__ == "__main__":
    main()
