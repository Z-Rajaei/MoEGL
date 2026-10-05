"""Compare the fastload cache against the old pyecore extractor on N models.

Standalone validation script (not part of the package): reimplements the old
pyecore-based _extract_resource and compares node/edge multisets per model.
Usage: python experiments/compare_pyecore_cache.py [n_models] [seed]
"""
import glob
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyecore.ecore import EAttribute, EEnum, EOrderedSet
from pyecore.resources import ResourceSet, URI


def _convert_attr(feature, value):
    if isinstance(feature.eType, EEnum):
        return getattr(value, "name", str(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _iter_objects(resource):
    for root in resource.contents:
        yield root
        yield from root.eAllContents()


def extract_old(path):
    rset = ResourceSet()
    resource = rset.get_resource(URI(path))
    objects = list(_iter_objects(resource))
    ids = {id(obj): i for i, obj in enumerate(objects)}
    nodes, edges = [], []
    for obj in objects:
        eclass = obj.eClass
        node = {"id": ids[id(obj)], "type": eclass.name, "attrs": {}}
        for f in eclass.eAllStructuralFeatures():
            if getattr(f, "derived", False):
                continue
            is_attr = isinstance(f, EAttribute)
            if is_attr:
                try:
                    if not obj.eIsSet(f):
                        continue
                except Exception:
                    continue
            try:
                value = obj.eGet(f)
            except Exception:
                continue
            if value is None:
                continue
            if is_attr:
                if f.many or isinstance(value, EOrderedSet):
                    node["attrs"][f.name] = [_convert_attr(f, v) for v in value]
                else:
                    node["attrs"][f.name] = _convert_attr(f, value)
            else:
                targets = value if (f.many or isinstance(value, EOrderedSet)) else [value]
                for tgt in targets:
                    tid = ids.get(id(tgt))
                    if tid is not None:
                        edges.append({"src": node["id"], "dst": tid, "type": f.name})
        # generic derivations (old extract._emf_generic_derivations)
        def add(rtype, target):
            tid = ids.get(id(target))
            if tid is not None and not any(
                e["src"] == node["id"] and e["dst"] == tid and e["type"] == rtype
                for e in edges
            ):
                edges.append({"src": node["id"], "dst": tid, "type": rtype})
        for g in getattr(obj, "eGenericSuperTypes", None) or []:
            cls = getattr(g, "eClassifier", None)
            if cls is not None:
                add("eSuperTypes", cls)
        g = getattr(obj, "eGenericType", None)
        if g is not None and getattr(obj, "eType", None) is None:
            cls = getattr(g, "eClassifier", None)
            if cls is not None:
                add("eType", cls)
        nodes.append(node)
    return {"nodes": nodes, "edges": edges}


def norm_attrs(attrs):
    # fastload keeps raw strings; pyecore typed values -> compare via str()
    return tuple(sorted(
        (k, tuple(str(v) for v in val) if isinstance(val, list) else str(val))
        for k, val in attrs.items()
    ))


def compare(old, new):
    """Return list of differences between pyecore model dict and fastload dict."""
    diffs = []
    if len(old["nodes"]) != len(new["nodes"]):
        diffs.append(f"node count: {len(old['nodes'])} != {len(new['nodes'])}")
    old_types = Counter(n["type"] for n in old["nodes"])
    new_types = Counter(n["type"] for n in new["nodes"])
    if old_types != new_types:
        diffs.append(f"node types: {old_types - new_types} / {new_types - old_types}")
    old_attrs = Counter(norm_attrs(n["attrs"]) for n in old["nodes"])
    new_attrs = Counter(norm_attrs(n["attrs"]) for n in new["nodes"])
    if old_attrs != new_attrs:
        diffs.append(f"attrs: only-old {list(old_attrs - new_attrs)[:3]} "
                     f"only-new {list(new_attrs - old_attrs)[:3]}")
    # edge multiset by (src type, edge type, dst type) -- ids may differ in order
    old_edges = Counter(
        (old["nodes"][e["src"]]["type"], e["type"], old["nodes"][e["dst"]]["type"])
        for e in old["edges"]
    )
    new_edges = Counter(
        (new["nodes"][e["src"]]["type"], e["type"], new["nodes"][e["dst"]]["type"])
        for e in new["edges"]
    )
    if old_edges != new_edges:
        diffs.append(f"edges: only-old {list(old_edges - new_edges)[:5]} "
                     f"only-new {list(new_edges - old_edges)[:5]}")
    return diffs


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    files = sorted(glob.glob("E:/Project/TCRMG-GNN-1.0.0/realModels/Ecore/*.ecore"))
    random.Random(seed).shuffle(files)
    files = files[:n]

    from moegl.config import Config
    import moegl.extract  # noqa: F401

    ex = sys.modules["moegl.extract"]  # package attr is shadowed by the function

    ok, bad = 0, []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        cfg = Config(format="ecore", metamodelpath="",
                     modelspath=os.path.dirname(path), output_format="NetworkX")
        # extract just this one file by monkey-narrowing the glob scope
        old_glob = ex.glob.glob
        try:
            ex.glob.glob = lambda p: [path]
            from moegl import extract as _extract
            cache = _extract(cfg)
        finally:
            ex.glob.glob = old_glob
        new_model = cache["models"][0] if cache["models"] else None
        try:
            old_model = extract_old(path)
        except Exception as exc:
            if new_model is None:
                continue
            bad.append((name, [f"pyecore failed: {exc}"]))
            continue
        if new_model is None:
            bad.append((name, ["fastload failed"]))
            continue
        d = compare(old_model, new_model)
        if d:
            bad.append((name, d))
        else:
            ok += 1
    print(f"equivalent: {ok}/{len(files)}")
    for name, d in bad:
        print(name, *d, sep="\n  ")


if __name__ == "__main__":
    main()
