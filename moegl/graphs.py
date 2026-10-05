"""Stage 2b: adapted ModelCache -> NetworkX MultiDiGraph per model."""

import networkx as nx

from .adapt import adapt
from .encoders import AttributeEncoder


def build_graphs(cache, config):
    """Return {model_name: nx.MultiDiGraph}.

    Node attribute ``type`` and edge attribute ``type`` are always present;
    declared attributes are encoded with the fitted encoders.
    """
    adapted = adapt(cache, config)
    encoder = AttributeEncoder(
        adapted["meta"].get("encodings", {}),
        w2v_dim=config.w2v_dim,
        seed=config.seed,
    ).fit(adapted["models"])

    graphs = {}
    for model in adapted["models"]:
        g = nx.MultiDiGraph()
        for node in model["nodes"]:
            attrs = {
                k: encoder.encode(node["type"], k, v)
                for k, v in node["attrs"].items()
            }
            g.add_node(node["id"], type=node["type"], **attrs)
        for edge in model["edges"]:
            if edge["src"] in g and edge["dst"] in g:
                g.add_edge(edge["src"], edge["dst"], type=edge["type"])
        graphs[model["name"]] = g
    return graphs


def encode(config):
    """Convenience: extract + build_graphs in one call."""
    from .extract import extract

    return build_graphs(extract(config), config)
