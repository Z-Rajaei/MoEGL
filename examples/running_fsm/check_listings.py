"""Run the four YAML listings printed in the paper. Paths are relative to the MoEGL package root."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from moegl import build_graphs, load_config
from moegl.extract import extract
from moegl.hetero import build_hetero

HERE = os.path.dirname(os.path.abspath(__file__))


def adjacency(g):
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    m = [[0] * len(nodes) for _ in nodes]
    for u, v in g.edges():
        m[idx[u]][idx[v]] = 1
    labels = [g.nodes[n].get("type") for n in nodes]
    return labels, m


def main():
    files = [
        "listing_schema.yaml",
        "listing_structure.yaml",
        "listing_attributes.yaml",
        "listing_metamodel.yaml",
    ]
    for name in files:
        cfg = load_config(os.path.join(HERE, name))
        print("LOADED", name, cfg.format, cfg.output_format, "attrs", cfg.include_all_attributes)

    cfg = load_config(os.path.join(HERE, "listing_structure.yaml"))
    graphs = build_graphs(extract(cfg), cfg)
    g = next(iter(graphs.values()))
    labels, m = adjacency(g)
    print("STRUCTURE labels", labels)
    print("STRUCTURE nodes", g.number_of_nodes(), "edges", g.number_of_edges())
    print("STRUCTURE edge types", sorted({d["type"] for _, _, d in g.edges(data=True)}))
    for row in m:
        print(" ".join(str(x) for x in row))
    loops = [(u, v, d["type"]) for u, v, d in g.edges(data=True) if u == v]
    print("SELF", loops)

    cfg = load_config(os.path.join(HERE, "listing_schema.yaml"))
    gs = build_graphs(extract(cfg), cfg)
    sg = next(iter(gs.values()))
    print("SCHEMA", sg.number_of_nodes(), sg.number_of_edges(),
          sorted({d["type"] for _, _, d in sg.edges(data=True)}))
    for n, d in sg.nodes(data=True):
        if d["type"] == "Transition":
            print("SCHEMA transition attrs", {k: d[k] for k in d if k != "type"})
            break

    cfg = load_config(os.path.join(HERE, "listing_attributes.yaml"))
    from moegl.adapt import adapt
    from moegl.encoders import AttributeEncoder

    adapted = adapt(extract(cfg), cfg)
    enc = AttributeEncoder(
        adapted["meta"].get("encodings", {}), w2v_dim=cfg.w2v_dim, seed=cfg.seed
    ).fit(adapted["models"])
    for node in adapted["models"][0]["nodes"]:
        if node["type"] == "State" and node["attrs"].get("timeout") == "1":
            print("STATE1 attrs", list(node["attrs"]))
            print("STATE1 timeout", node["attrs"]["timeout"])
            print("STATE1 onehot", enc.encode("State", "type", node["attrs"]["type"]))
            print("STATE1 name dim", len(enc.encode("State", "name", node["attrs"]["name"])))

    data = build_hetero(extract(cfg), cfg)
    h = next(iter(data.values()))
    print("HETERO")
    print(h)

    cfg = load_config(os.path.join(HERE, "listing_metamodel.yaml"))
    mg = build_graphs(extract(cfg), cfg)
    gg = next(iter(mg.values()))
    print("META nodes", sorted({d["type"] for _, d in gg.nodes(data=True)}))
    print("META edges", sorted({d["type"] for _, _, d in gg.edges(data=True)}))
    shown = 0
    for _, d in gg.nodes(data=True):
        if d["type"] == "EClass":
            print("ECLASS", {k: (len(v) if isinstance(v, list) else v) for k, v in d.items()})
            shown += 1
            if shown >= 2:
                break
    data = build_hetero(extract(cfg), cfg)
    print("META HETERO")
    print(next(iter(data.values())))


if __name__ == "__main__":
    main()
