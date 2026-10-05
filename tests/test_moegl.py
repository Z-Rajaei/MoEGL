import os

import pytest
import yaml

from moegl import build_graphs, extract, load_config

HERE = os.path.dirname(__file__)
FSM_DIR = os.path.join(HERE, "data", "fsm")
TINY_DIR = os.path.join(HERE, "data", "tiny")


def make_config(tmp_path, classes_yaml, fmt="ecore", modelspath=TINY_DIR,
                metamodelpath="", out="NetworkX"):
    doc = {
        "input": {
            "format": fmt,
            "metamodelpath": metamodelpath,
            "modelspath": modelspath,
        },
        "output": {"format": out},
        "adaptations": {
            "metamodels": {
                "packages": {
                    "pkg": {"uri": {"u": {"classes": classes_yaml}}}
                }
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(doc))
    return load_config(str(path))


def edge_types(g):
    return {d["type"] for _, _, d in g.edges(data=True)}


def node_types(g):
    return {d["type"] for _, d in g.nodes(data=True)}


# 1 --------------------------------------------------------------------
def test_fsm_exclude_rename(tmp_path):
    cfg = make_config(
        tmp_path,
        fmt="xmi",
        modelspath=FSM_DIR,
        metamodelpath=os.path.join(FSM_DIR, "statecharts.ecore"),
        classes_yaml={
            "exclude": ["Action"],
            "StateMachine": {
                "features": {
                    "transitions": {"renaming": "trans"},
                    "top": {"renaming": "trans"},
                }
            },
        },
    )
    graphs = build_graphs(extract(cfg), cfg)
    g = graphs["statecharts_example"]
    assert "Action" not in node_types(g)
    assert "actions" not in edge_types(g)
    trans = [d for _, _, d in g.edges(data=True) if d["type"] == "trans"]
    assert len(trans) > 0


# 2 --------------------------------------------------------------------
def test_include_subtype(tmp_path):
    cfg = make_config(tmp_path, classes_yaml={"include": ["EClassifier"]})
    graphs = build_graphs(extract(cfg), cfg)
    types = node_types(graphs["tiny1"])
    assert {"EClass", "EDataType", "EEnum"} <= types
    assert "EPackage" not in types
    assert "EAttribute" not in types


# 3 --------------------------------------------------------------------
def test_exclude_priority(tmp_path):
    cfg = make_config(
        tmp_path,
        classes_yaml={"include": ["EClass", "EDataType"],
                      "exclude": ["EDataType"]},
    )
    graphs = build_graphs(extract(cfg), cfg)
    types = node_types(graphs["tiny1"])
    assert "EClass" in types
    assert "EDataType" not in types
    assert types == {"EClass"}


# 4 --------------------------------------------------------------------
def test_attribute_rules(tmp_path):
    cfg_off = make_config(
        tmp_path, classes_yaml={"includeAllAttributes": False}
    )
    g = build_graphs(extract(cfg_off), cfg_off)["tiny1"]
    for _, d in g.nodes(data=True):
        attrs = {k: v for k, v in d.items() if k != "type"}
        assert attrs == {}

    cfg_on = make_config(
        tmp_path, classes_yaml={"includeAllAttributes": True}
    )
    g = build_graphs(extract(cfg_on), cfg_on)["tiny1"]
    names = [d.get("name") for _, d in g.nodes(data=True) if "name" in d]
    assert "tiny1" in names and "Foo" in names

    cfg_excl = make_config(
        tmp_path,
        classes_yaml={
            "includeAllAttributes": True,
            "EClass": {"exclude": ["name"]},
        },
    )
    g = build_graphs(extract(cfg_excl), cfg_excl)["tiny1"]
    for nid, d in g.nodes(data=True):
        if d["type"] == "EClass":
            assert "name" not in d
        if d["type"] == "EPackage":
            assert d.get("name") == "tiny1"


# 5 --------------------------------------------------------------------
def test_encoders(tmp_path):
    cfg = make_config(
        tmp_path,
        classes_yaml={
            "includeAllAttributes": True,
            "EClass": {"features": {"name": {"encoding": "one-hot"}}},
            "EPackage": {"features": {"name": {"encoding": "word2vec"}}},
        },
    )
    graphs = build_graphs(extract(cfg), cfg)

    # one-hot: a vector with a single 1
    onehot = [
        d["name"]
        for g in graphs.values()
        for _, d in g.nodes(data=True)
        if d["type"] == "EClass" and "name" in d
    ]
    assert onehot
    for vec in onehot:
        assert sum(1 for x in vec if x == 1.0) == 1
        assert sum(vec) == 1.0

    # word2vec: fixed-size vector; identical strings -> identical vectors
    pkgs = {
        gname: next(
            d["name"]
            for _, d in g.nodes(data=True)
            if d["type"] == "EPackage" and "name" in d
        )
        for gname, g in graphs.items()
    }
    v1, v2 = pkgs["tiny1"], pkgs["tiny2"]
    assert len(v1) == len(v2) == 64

    cfg2 = make_config(
        tmp_path,
        classes_yaml={
            "includeAllAttributes": True,
            "EPackage": {"features": {"name": {"encoding": "word2vec"}}},
        },
    )
    cache = extract(cfg2)
    # fabricate two models sharing an identical attribute value
    import copy

    m = copy.deepcopy(cache["models"][0])
    m["name"] = "copy_of_tiny1"
    cache["models"].append(m)
    graphs2 = build_graphs(cache, cfg2)
    a = next(
        d["name"]
        for _, d in graphs2["tiny1"].nodes(data=True)
        if d["type"] == "EPackage" and "name" in d
    )
    b = next(
        d["name"]
        for _, d in graphs2["copy_of_tiny1"].nodes(data=True)
        if d["type"] == "EPackage" and "name" in d
    )
    assert a == b


def test_paper_fsm_listings_have_no_self_loop():
    """The running-example listings printed in the paper load, and Matrix 1
    has a 0 in the FSM diagonal: that class has no reference to itself."""
    root = os.path.dirname(HERE)
    prev = os.getcwd()
    os.chdir(root)
    try:
        for name in (
            "listing_schema.yaml",
            "listing_structure.yaml",
            "listing_attributes.yaml",
            "listing_metamodel.yaml",
        ):
            load_config(os.path.join(root, "examples", "running_fsm", name))
        cfg = load_config(
            os.path.join(root, "examples", "running_fsm", "listing_structure.yaml")
        )
        g = next(iter(build_graphs(extract(cfg), cfg).values()))
    finally:
        os.chdir(prev)
    nodes = list(g.nodes())
    assert [g.nodes[n]["type"] for n in nodes] == [
        "FSM", "Transition", "Transition", "Transition",
        "State", "State", "State", "State",
    ]
    idx = {n: i for i, n in enumerate(nodes)}
    m = [[0] * 8 for _ in range(8)]
    for u, v in g.edges():
        m[idx[u]][idx[v]] = 1
    assert m[0] == [0, 1, 1, 1, 1, 1, 1, 1]
    assert all(m[i][i] == 0 for i in range(8))
    assert g.number_of_edges() == 13
    assert sorted({d["type"] for _, _, d in g.edges(data=True)}) == [
        "source", "states", "target", "trans",
    ]


# 6 --------------------------------------------------------------------
def test_hetero_roundtrip(tmp_path):
    pytest.importorskip("torch_geometric")
    from moegl import build_hetero

    cfg = make_config(tmp_path, classes_yaml={"include": ["EClassifier"]})
    cache = extract(cfg)
    graphs = build_graphs(cache, cfg)
    datas = build_hetero(cache, cfg)

    assert set(datas) == set(graphs)
    for name, g in graphs.items():
        data = datas[name]
        nx_counts = {}
        for _, d in g.nodes(data=True):
            nx_counts[d["type"]] = nx_counts.get(d["type"], 0) + 1
        for ntype, count in nx_counts.items():
            assert data[ntype].x.shape[0] == count
            assert data[ntype].node_id.shape[0] == count
            assert data[ntype].orig_id.shape[0] == count
        total_edges = sum(
            data[et].edge_index.shape[1] for et in data.edge_types
        )
        assert total_edges == g.number_of_edges()
