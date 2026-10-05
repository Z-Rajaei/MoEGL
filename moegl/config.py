"""YAML configuration loading and validation for MoEGL v2."""

import os
from dataclasses import dataclass, field

import yaml

VALID_FORMATS = ("ecore", "xmi")
VALID_OUTPUTS = ("NetworkX", "HPyG")
VALID_ENCODINGS = ("rawstring", "one-hot", "word2vec", "worde4mde")

# keys of the `classes` section that are not per-class sections
_CLASS_RESERVED = {
    "include",
    "exclude",
    "includeAllAttributes",
    "excludeAllAttributes",
    "includeAttributes",
}
# keys of a per-class section that are not feature shorthands
_RULE_RESERVED = {"renaming", "include", "exclude", "features"}


@dataclass
class FeatureRule:
    renaming: str | None = None
    encoding: str | None = None


@dataclass
class ClassRule:
    renaming: str | None = None
    include: list | None = None
    exclude: list = field(default_factory=list)
    features: dict = field(default_factory=dict)  # feature name -> FeatureRule


@dataclass
class Config:
    format: str
    metamodelpath: str
    modelspath: str
    output_format: str
    extension: str | None = None  # model file extension, e.g. ".sct" (default: .ecore/.xmi by format)
    class_include: list | None = None
    class_exclude: list = field(default_factory=list)
    include_all_attributes: bool = False
    class_rules: dict = field(default_factory=dict)  # class name -> ClassRule
    w2v_dim: int = 64
    seed: int = 42


def _norm_encoding(value, where):
    if value is None:
        return None
    enc = str(value).strip()
    low = enc.lower()
    if low in ("rawstring", "raw"):
        return "rawstring"
    if low in ("one-hot", "onehot"):
        return "one-hot"
    if low == "word2vec":
        return "word2vec"
    if low in ("worde4mde", "word2mde"):
        return "worde4mde"
    raise ValueError(
        f"Unknown encoding '{enc}' at {where}; supported: "
        "RawString, one-hot, word2vec, worde4mde"
    )


def _parse_classes(classes):
    """Split a `classes:` mapping into global filters and per-class rules."""
    class_include = classes.get("include")
    class_exclude = list(classes.get("exclude") or [])
    include_all = bool(
        classes.get("includeAllAttributes", classes.get("includeAttributes", False))
    )
    if classes.get("excludeAllAttributes"):
        include_all = False

    rules = {}
    for name, section in classes.items():
        if name in _CLASS_RESERVED:
            continue
        section = section or {}
        if not isinstance(section, dict):
            raise ValueError(f"Class section '{name}' must be a mapping")
        rule = ClassRule(
            renaming=section.get("renaming"),
            include=list(section["include"]) if section.get("include") is not None else None,
            exclude=list(section.get("exclude") or []),
        )
        features = dict(section.get("features") or {})
        # the paper's listings also allow features directly under the class
        for k, v in section.items():
            if k not in _RULE_RESERVED and isinstance(v, dict):
                features.setdefault(k, v)
        for fname, fsec in features.items():
            fsec = fsec or {}
            rule.features[fname] = FeatureRule(
                renaming=fsec.get("renaming"),
                encoding=_norm_encoding(
                    fsec.get("encoding"), f"classes.{name}.features.{fname}"
                ),
            )
        rules[name] = rule
    return class_include, class_exclude, include_all, rules


def _find_classes_sections(adaptations):
    """Yield every `classes` mapping under adaptations.metamodels.packages.*.uri.*."""
    packages = ((adaptations or {}).get("metamodels") or {}).get("packages") or {}
    for _pkg_name, pkg in packages.items():
        for _uri, uri_sec in (pkg.get("uri") or {}).items():
            classes = (uri_sec or {}).get("classes")
            if classes is not None:
                yield classes


def load_config(path):
    """Load a MoEGL YAML file into a validated :class:`Config`."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Empty or invalid config file: {path}")

    inp = raw.get("input") or raw.get("inputmodels") or {}
    fmt = str(inp.get("format", "")).lower()
    if fmt not in VALID_FORMATS:
        raise ValueError(f"input.format must be one of {VALID_FORMATS}, got '{fmt}'")
    modelspath = inp.get("modelspath")
    if not modelspath:
        raise ValueError("input.modelspath is required")
    metamodelpath = inp.get("metamodelpath") or ""
    extension = inp.get("extension") or None

    out = raw.get("output") or {}
    if isinstance(out, str):
        out_fmt = out
    else:
        out_fmt = out.get("format", "NetworkX")
    if str(out_fmt).lower() in ("hpyg", "pyg", "heterodata"):
        out_fmt = "HPyG"
    elif str(out_fmt).lower() in ("networkx", "nx"):
        out_fmt = "NetworkX"
    else:
        raise ValueError(
            f"output.format must be one of {VALID_OUTPUTS}, got '{out_fmt}'"
        )

    class_include, class_exclude, include_all, rules = None, [], False, {}
    for classes in _find_classes_sections(raw.get("adaptations")):
        inc, exc, ia, r = _parse_classes(classes)
        if inc is not None:
            class_include = (class_include or []) + list(inc)
        class_exclude += exc
        include_all = include_all or ia
        for name, rule in r.items():
            rules[name] = rule

    w2v_dim = int(raw.get("word2vec_dim", 64))
    seed = int(raw.get("seed", 42))

    return Config(
        format=fmt,
        metamodelpath=str(metamodelpath),
        modelspath=str(modelspath),
        output_format=out_fmt,
        extension=str(extension) if extension else None,
        class_include=class_include,
        class_exclude=class_exclude,
        include_all_attributes=include_all,
        class_rules=rules,
        w2v_dim=w2v_dim,
        seed=seed,
    )
