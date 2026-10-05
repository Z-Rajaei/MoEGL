# MoEGL v2

Model-to-graph encoder driven by a YAML DSL. Two stages:

1. **Stage 1 – extract**: models (`.ecore` or `.xmi`) are read with an lxml
   loader (`moegl/fastload.py`; see `docs/loader.md`) and stored in a
   plain-dict `ModelCache` (picklable + JSON-serialisable), so the traversal
   is done once. An earlier PyEcore loader is not the timed tool.
2. **Stage 2 – build graphs**: the DSL `adaptations` (include / exclude /
   renaming / encodings) are applied to the cache, attribute encoders are
   fitted once over the whole dataset, and one graph per model is materialised
   as `networkx.MultiDiGraph` or `torch_geometric.HeteroData` (lazy import —
   the core package has no torch dependency).

## Install

```
pip install -r requirements.txt          # lxml, networkx, pyyaml, gensim, numpy
pip install torch torch_geometric        # only for output.format: HPyG
```

## DSL (YAML)

```yaml
input:
  format: "ecore" | "xmi"        # ecore = the models ARE metamodels (Ecore instances)
  metamodelpath: "<path to .ecore>"   # optional for format=ecore (built-in Ecore is used)
  modelspath: "<dir with *.ecore | *.xmi>"
output:
  format: "NetworkX" | "HPyG"
adaptations:
  metamodels:
    packages:
      <PackageName>:
        uri:
          <nsURI>:
            classes:
              include: [ClassA, ClassB]       # optional whitelist of EClass names
              exclude: [ClassC]               # optional blacklist, wins over include
              includeAllAttributes: true|false  # default false -> no attribute values
              <ClassName>:                    # per-class section
                renaming: "NewName"           # renames the node type
                include: [feat1, feat2]       # whitelist of features (attrs AND refs)
                exclude: [feat3]              # blacklist, wins over include
                features:
                  <featureName>:
                    renaming: "newName"       # renames edge type / attribute key
                    encoding: "RawString"|"one-hot"|"word2vec"|"worde4mde"
```

### Semantics

* **Class filtering**: `exclude` wins over `include`; when `include` is absent
  every class is included. Matching is subtype-aware: a node of EClass `C`
  matches a list when `C` or any of its transitive `eSuperTypes` appears in it.
* **Feature filtering**: per class, `exclude` wins over `include`. When no
  applicable class section defines an `include` list, all references are kept
  and attributes are kept only if `includeAllAttributes: true`. When at least
  one applicable section defines `include`, only listed features are kept —
  the effective `include`/`exclude` are the **union** over the node's own
  class section and all its supertype sections (a rule for `ETypedElement`
  applies to `EAttribute` and `EReference` nodes).
* `renaming` of a class is taken from the most specific applicable section
  that defines it; feature renaming likewise, and applies to edge types and
  attribute keys.
* Removing a class removes its nodes and every edge touching them; removing a
  feature removes only the edge/attribute.
* Derived references (`feature.derived`) are never encoded.
* Node ids: `0..n-1` per model in document order, root first.
  An edge is `(src, dst, type=<reference name>)`. At most one edge is kept
  for the same source, target and feature name (`MultiDiGraph`).
* A reverse edge is written when the feature records an opposite and that
  opposite feature exists and is not derived. On the bundled Ecore
  metamodel, `EReference.eOpposite` records no opposite, so that edge is
  written only where the file serialises it.
* HeteroData stores node features and `edge_index` only, grouped by
  `(source type, reference name, target type)`. Multiplicity and containment
  stay on the declaring node. There is no edge-feature tensor.

### Encodings

* `RawString` (default): the value is kept as-is.
* `one-hot`: vocabulary built from every model passed to `fit`. An unseen
  value uses a dedicated last slot.
* `word2vec`: gensim Word2Vec (`seed=42`, `workers=1`, `sg=1`, dim 64,
  window 3, `min_count=1`, 20 epochs) trained on the values of every model
  passed to `fit`, tokenised on camelCase/snake_case/digits. The value
  embedding is the mean of its token embeddings. An unknown token stays the
  zero vector. A supervised split must call `fit` on the training fold only;
  fitting the whole corpus before the split puts test names into the vectors.
  Encoding a set that has no held-out split still fits on that whole set.
* `worde4mde`: loads the frozen `sgram-mde` vectors through the `worde4mde`
  package and does not train them. Raises `ImportError` at fit time if the
  package is not installed.

## CLI

```
python -m moegl config.yaml --cache CACHE_DIR --out OUT_DIR [--stage extract|graphs|all]
```

* `extract` writes `CACHE_DIR/cache.pkl` + `cache.json`.
* `graphs` reads the cache and writes `OUT_DIR/<model>.json` (node-link JSON)
  or `OUT_DIR/<model>.pt` for HPyG.
* Prints a summary: `#models, #skipped, avg nodes, avg edges, wall time per stage`.

## Python API

```python
from moegl import load_config, extract, build_graphs, build_hetero, encode

cfg = load_config("config.yaml")
cache = extract(cfg)                 # Stage 1
graphs = build_graphs(cache, cfg)    # Stage 2 -> dict[name, nx.MultiDiGraph]
# datas = build_hetero(cache, cfg)  # Stage 2 -> dict[name, HeteroData]
# graphs = encode(cfg)              # both stages
```

`ModelCache` schema:

```
{"models": [{"name": str,
             "nodes": [{"id": int, "type": str, "attrs": {k: v}}],
             "edges": [{"src": int, "dst": int, "type": str}]}],
 "meta": {"format": ..., "n_models": ..., "skipped": [[file, error]],
          "supertypes": {class: [supertypes]}}}
```

Files the loader cannot parse are skipped and recorded in `meta["skipped"]`.
The printed `avg_nodes` / `avg_edges` are the cache counts, before stage 2
writes the adapted graphs.

## Tests

```
pytest tests/    # tests 1-5 need the base venv; test 6 needs torch_geometric
```

## Running example

`examples/running_fsm` holds the metamodel, the instance and the four YAML
listings from the paper. `examples/running_fsm/check_listings.py` is the
script that executed them.

## Reproducing the paper's experiments (`experiments/`)

`configs/` holds the configurations: `exp1_lopez_ecore.yaml`,
`exp1_lopez_rds.yaml`, `exp1_lopez_yakindu.yaml` (López et al.),
`exp2_rahimi_carwash.yaml`, `exp3_miranda_fsm.yaml`, and
`exp1_alt_no_datatypes.yaml` (re-encoding timing). Further YAML files under
`configs/` are the classification setups.

The 500 Ecore models and the published graphs are not in this repository.
They come from the TCRMG-GNN replication package
(<https://github.com/Antolin1/TCRMG-GNN>, release 1.0.0:
`realModels/Ecore`, `realGraphs/Ecore/all`). Set those paths at the top of
`experiments/run_experiments.py` and in the configs. ModelSet is likewise
external and is not bundled here.

```
# fidelity, Table 1 statistics and timing
python experiments/run_experiments.py --runs 10
# lines of code with cloc (npm install cloc)
python experiments/count_loc.py
```

The Java encoder of Experiment 1 is the authors' code, run through
`experiments/lopez_driver`. The copies used for the line count are in
`experiments/bespoke_encoders/` (see `SOURCES.md` there).

Manuscript numbers are read from these files, not from an earlier session:

| File | What it records |
| --- | --- |
| `results/fidelity_cert.json` | 500/500 Ecore, 500/500 RDS, 369/369 Yakindu |
| `results/timing_paired_ecore_final.json` | ten fresh processes, MoEGL 1.11±0.02 s, Java 1.48±0.10 s |
| `results/memory_paired_ecore.json` | peak memory of the same two commands, separate session |
| `results/exp4_v2.json` | classification in Section 6.6 |
| `results/exp4_convergence.json` | homogeneous GIN and heterogeneous RGCN training curves |
| `results/loc.json` | line counts |

`results/exp4_classify.json` is an earlier run and is not a table in the
manuscript.

### Loader notes (Stage 1)

The loader strips XML comments while parsing. A fragment that still resolves
after a rename is kept and the URI is recorded in `meta["repaired"]`.
Absolute, `platform:`, `pathmap:` and a relative URI whose file exists beside
the model create no node and no edge. For Ecore models, `eSuperTypes` is
taken from `eGenericSuperTypes` and `eType` from `eGenericType.eClassifier`
when the file does not already carry that edge. The details are in
`docs/loader.md`.
