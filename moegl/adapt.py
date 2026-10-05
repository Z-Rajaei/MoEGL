"""Stage 2a: apply the include/exclude/renaming rules of the DSL to a ModelCache.

Semantics (see README / spec):
* Class filtering: `exclude` wins over `include`; when `include` is absent every
  class is included. Matching is subtype-aware: a node of EClass C matches a
  list when C or any of its transitive eSuperTypes appears in it.
* Feature filtering: per class, `exclude` wins over `include`; when no
  applicable class section defines an `include` list all references are kept,
  and attributes are kept only if `includeAllAttributes` is true.
  Rules written for a supertype apply to its subtypes: the effective `include`
  is the union of the include lists of all applicable class sections, likewise
  for `exclude`.
* `renaming` for a node type is taken from the most specific applicable class
  section that defines it; feature renaming likewise.
* Removing a class removes its nodes and every edge touching them; removing a
  feature removes only the edge/attribute.
"""

from .config import Config


def _supers(node_type, supertypes):
    return supertypes.get(node_type, [])


def _class_kept(node_type, config: Config, supertypes):
    names = [node_type] + list(_supers(node_type, supertypes))
    if any(n in config.class_exclude for n in names):
        return False
    if config.class_include is None:
        return True
    return any(n in config.class_include for n in names)


def _applicable_rules(node_type, config: Config, supertypes):
    """ClassRule objects applicable to a node, most specific first."""
    rules = []
    for name in [node_type] + list(_supers(node_type, supertypes)):
        rule = config.class_rules.get(name)
        if rule is not None:
            rules.append(rule)
    return rules


def _effective_lists(rules):
    include, exclude = set(), set()
    has_include = False
    for r in rules:
        if r.include is not None:
            has_include = True
            include.update(r.include)
        exclude.update(r.exclude)
    return (include if has_include else None), exclude


def _class_renaming(node_type, rules):
    for r in rules:  # most specific first
        if r.renaming:
            return r.renaming
    return node_type


def _feature_rule(rules, feat):
    for r in rules:  # most specific first
        fr = r.features.get(feat)
        if fr is not None:
            return fr
    return None


def adapt(cache, config: Config):
    """Return a new ModelCache with the DSL adaptations applied."""
    supertypes = (cache.get("meta") or {}).get("supertypes", {})
    encodings = {}  # (node type, renamed feature) -> encoding, for Stage 2b

    out_models = []
    for model in cache.get("models", []):
        kept_nodes, kept_ids = {}, set()
        for node in model.get("nodes", []):
            ntype = node["type"]
            if not _class_kept(ntype, config, supertypes):
                continue
            rules = _applicable_rules(ntype, config, supertypes)
            include, exclude = _effective_lists(rules)
            new_type = _class_renaming(ntype, rules)

            attrs = {}
            for key, value in node.get("attrs", {}).items():
                if key in exclude:
                    continue
                if include is not None:
                    if key not in include:
                        continue
                elif not config.include_all_attributes:
                    continue
                fr = _feature_rule(rules, key)
                new_key = fr.renaming if fr and fr.renaming else key
                if fr and fr.encoding and fr.encoding != "rawstring":
                    encodings[(new_type, new_key)] = fr.encoding
                attrs[new_key] = value

            new_node = {
                "id": node["id"],
                "type": new_type,
                "attrs": attrs,
                "_rules": rules,  # internal, stripped before output
            }
            kept_nodes[node["id"]] = new_node
            kept_ids.add(node["id"])

        edges = []
        for edge in model.get("edges", []):
            if edge["src"] not in kept_ids or edge["dst"] not in kept_ids:
                continue
            src_node = kept_nodes[edge["src"]]
            include, exclude = _effective_lists(src_node["_rules"])
            etype = edge["type"]
            if etype in exclude:
                continue
            if include is not None and etype not in include:
                continue
            fr = _feature_rule(src_node["_rules"], etype)
            edges.append(
                {
                    "src": edge["src"],
                    "dst": edge["dst"],
                    "type": (fr.renaming if fr and fr.renaming else etype),
                }
            )

        nodes = [
            {"id": n["id"], "type": n["type"], "attrs": n["attrs"]}
            for n in model.get("nodes", [])
            if n["id"] in kept_ids
        ]
        for n in nodes:
            n["type"] = kept_nodes[n["id"]]["type"]
            n["attrs"] = kept_nodes[n["id"]]["attrs"]

        out_models.append({"name": model["name"], "nodes": nodes, "edges": edges})

    meta = dict(cache.get("meta") or {})
    meta["encodings"] = encodings
    return {"models": out_models, "meta": meta}
