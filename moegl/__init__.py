"""MoEGL v2 -- model-to-graph encoder driven by a YAML DSL."""

from .config import load_config
from .extract import extract
from .graphs import build_graphs, encode

__all__ = ["load_config", "extract", "build_graphs", "build_hetero", "encode"]


def build_hetero(cache, config):
    from .hetero import build_hetero as _impl

    return _impl(cache, config)
