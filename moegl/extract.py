"""Stage 1: load models with the fast lxml loader and build a ModelCache.

ModelCache = {
  "models": [{"name": str,
              "nodes": [{"id": int, "type": str, "attrs": {k: v}}],
              "edges": [{"src": int, "dst": int, "type": str}]}],
  "meta": {"format": str, "n_models": int,
           "skipped": [[file, error], ...],
           "repaired": [[file, desc], ...],
           "supertypes": {class_name: [supertype names]}}
}
Everything in the cache is picklable and JSON-serialisable.
"""

import glob
import json
import os
import pickle

from .fastload import BUNDLED_ECORE, metamodel_index, parse_model, transitive_supertypes


def extract(config):
    """STAGE 1: models -> ModelCache."""
    # metamodel index: the bundled Ecore.ecore (for format="ecore" and for
    # ecore:-prefixed xsi:types in xmi models) plus input.metamodelpath, which
    # may be a single .ecore file or a directory indexed wholesale
    index = metamodel_index(BUNDLED_ECORE)
    if config.metamodelpath:
        extra = metamodel_index(config.metamodelpath)
        for ns_uri, pkg in extra.items():
            if ns_uri in index:
                index[ns_uri]["classes"].update(pkg["classes"])
            else:
                index[ns_uri] = pkg

    ext = config.extension or (".ecore" if config.format == "ecore" else ".xmi")
    files = sorted(glob.glob(os.path.join(config.modelspath, "*" + ext)))

    supers_of = transitive_supertypes(index)
    models, skipped, repaired, seen_types = [], [], [], set()
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            model, repair = parse_model(path, index, config.format)
            model["name"] = name
            repaired += repair
            for node in model["nodes"]:
                seen_types.add(node["type"])
            models.append(model)
        except Exception as exc:  # record, never silently drop
            skipped.append([os.path.basename(path), f"{type(exc).__name__}: {exc}"])

    supertypes = {t: supers_of(t) for t in sorted(seen_types)}
    return {
        "models": models,
        "meta": {
            "format": config.format,
            "n_models": len(models),
            "skipped": skipped,
            "repaired": repaired,
            "supertypes": supertypes,
        },
    }


def save_cache(cache, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    pkl_path = os.path.join(cache_dir, "cache.pkl")
    json_path = os.path.join(cache_dir, "cache.json")
    with open(pkl_path, "wb") as f:
        pickle.dump(cache, f)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    return pkl_path, json_path


def load_cache(cache_dir):
    with open(os.path.join(cache_dir, "cache.pkl"), "rb") as f:
        return pickle.load(f)
