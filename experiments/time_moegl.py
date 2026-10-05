"""One timing sample of MoEGL in a fresh process.

Usage: python time_moegl.py <config.yaml>
Prints: MOEGL_STAGE1_MS <ms>   (extraction: pyecore loading + ModelCache)
        MOEGL_STAGE2_MS <ms>   (adaptation + encoding + NetworkX graph construction)
        MOEGL_MODELS <n> NODES <n> EDGES <n>
"""
import os
import sys
import time

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from moegl import load_config, extract, build_graphs  # noqa: E402


def main():
    config = load_config(sys.argv[1])
    t0 = time.perf_counter()
    cache = extract(config)
    t1 = time.perf_counter()
    graphs = build_graphs(cache, config)
    t2 = time.perf_counter()
    nodes = sum(g.number_of_nodes() for g in graphs.values())
    edges = sum(g.number_of_edges() for g in graphs.values())
    print(f"MOEGL_STAGE1_MS {(t1 - t0) * 1000:.3f}")
    print(f"MOEGL_STAGE2_MS {(t2 - t1) * 1000:.3f}")
    print(f"MOEGL_MODELS {len(graphs)} NODES {nodes} EDGES {edges}")


if __name__ == "__main__":
    main()
