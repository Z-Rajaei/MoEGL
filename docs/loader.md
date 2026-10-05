# MoEGL fast XMI/Ecore loader — specification

Stage-1 model loading must **not** use pyecore (it is a pure-Python EMF
re-implementation and is the dominant cost of the pipeline: ~15 ms/model).
`moegl/fastload.py` replaces it with a direct `lxml.etree` parser that
reproduces the *observable* semantics of EMF's `XMIResourceImpl` + the
`getAllContents()` traversal used by the reference encoders.

The loader produces the same `ModelCache` dict as `extract.py` today:

```python
{"models": [{"name": str,
             "nodes": [{"id": int, "type": str, "attrs": {str: value}}],
             "edges": [{"src": int, "dst": int, "type": str}]}],
 "meta": {"format": ..., "n_models": int, "skipped": [[file, err]],
          "repaired": [[file, desc]], "supertypes": {cls: [supers]}}}
```

## Metamodel index (`metamodel_index`)

Built once from one or more `.ecore` files (a directory may be given — all
`*.ecore` inside are indexed). The bundled `metamodels/ecore/Ecore.ecore`
(extracted from `org.eclipse.emf.ecore-2.20.0.jar`) is used for
`format: "ecore"` datasets, i.e. datasets whose files are themselves
metamodels.

A `.ecore` file is parsed directly (fixed, simple structure — no need for the
general model parser): walk `eClassifiers` elements at any depth; each has
`xsi:type="ecore:EClass|ecore:EEnum|ecore:EDataType"` and `name`; EClasses have
`eStructuralFeatures` children (`xsi:type="ecore:EReference|ecore:EAttribute"`,
`name`, `eType`, `upperBound`, `containment`, `eOpposite`, `eGenericType`/
`eTypeParameters` children). Build:

```
{nsURI: {"prefix": nsPrefix,
         "classes": {class_name: {
             "abstract": bool,
             "supertypes": [class names, resolved within the same file],
             "features": {feature_name: {"kind": "attr"|"ref",
                                         "type": type name or None,
                                         "many": bool,       # upperBound != 1
                                         "containment": bool}}}}}}
```

- `eSuperTypes` attribute values are `#//X` or `#//pkg/X` fragments (or a
  space-separated list); `eGenericSuperTypes` child elements with
  `eClassifier="#//X"` are an alternative serialization — collect both, dedup.
- `eType` attribute values may be `ecore:EDataType http://...#//EString`
  (external/built-in → feature "type" = the fragment's last segment, e.g.
  `EString`, flagged builtin), `#//X` (local), or absent (unresolved).
- `eAllFeatures(class)`: own features + supertypes' recursively (for feature
  lookup by name).

## Model parsing (`parse_model(path, index) -> model dict`)

1. Parse with `lxml.etree.XMLParser(remove_comments=True, recover=False)` —
   XML comments are thus handled transparently (this replaces the old
   "strip comments" repair).
2. Root: single element whose QName is `prefix:Class` (e.g.
   `<ecore:EPackage>`, `<com.genmymodel.rds.core:Database>`,
   `<sgraph:Statechart>`), or an `<xmi:XMI>` wrapper whose children are the
   roots. Resolve prefix → nsURI → class in the index. If the nsURI is not in
   the index, record in `skipped` and continue to the next file.
3. Walk the tree in document order (DFS). Each element is an object:
   - concrete class = `xsi:type` value (`prefix:Name`) if present, else the
     declared type of the containment feature it was reached through (for the
     root: the QName's class).
   - `id` = DFS position (0-based), matching EMF `getAllContents` order.
   - `type` = simple class name (after `:`).
4. Index for reference resolution, built in the same pass:
   - `xmi:id` / `xmi:ID` attribute → object.
   - name-fragment `#//a/b/...`: for objects with a `name` attribute, the EMF
     fragment is `//` + the `name` values along the containment path
     (`#//Pkg/Class/feature`). Register every named object.
   - positional fragment `#/i` (i-th root) and `#/i/@feat.j` / `#/@feat.j`
     (j-th value of feature `feat`). Register the containment-path form
     `#/0/@eClassifiers.3/@eStructuralFeatures.1` for every object.
5. Features of each object (lookup by name through `eAllFeatures`):
   - XML attributes other than `xmi:*`, `xsi:*`, `xmlns:*`:
     feature `attr` → literal value kept in `attrs` (raw string);
     feature `ref`  → value tokenized on whitespace; each token resolved.
   - Child elements: feature name = tag local name. The element is
     a **containment** (recurse into it as an object of the feature's type or
     `xsi:type`) unless it carries `href` — then it is a reference stub:
     resolve `href` and do not create a node.
   - Reference token resolution:
     - `#//a/b`, `#/@f.j`, `#/i/...`, bare `xmi:id` → internal lookup; if found
       → edge `(src, dst, feature_name)`; if the fragment is unknown → drop
       (unresolved proxy).
     - `anything.ext#//x`, `anything.ext#/i/...`, `platform:/...`,
       `pathmap://...`, `http(s)://...` → cross-resource:
       * if `anything.ext` exists relative to the model file → still external
         (out of scope: graphs only contain same-resource nodes — verified
         against the published graphs) → drop;
       * if it does **not** exist on disk AND its basename looks like a model
         file that was renamed (i.e. the fragment resolves when treated as
         local) → treat as **local**, resolve, and record
         `["<file>", "self-references to <anything.ext> made local"]` in
         `meta["repaired"]`;
       * otherwise → drop (proxy, as EMF does when the resource cannot be
         resolved — the reference encoders skip `eIsProxy()` objects).
     - Tokens of the form `prefix:Type uri#//frag` (the Ecore serialization of
       typed references, e.g. `eType="ecore:EDataType
       http://www.eclipse.org/emf/2002/Ecore#//EString"`): resolve the URI part
       with the same rules (that URI is external → dropped).
   - Only same-document targets produce edges. External metamodel objects
     (e.g. `EString`) must produce **no node and no edge** — verified: in the
     published Ecore graphs every `EDataType` node has an `ePackage` edge to
     the model's own single `EPackage` node.
6. EMF derivations specific to the Ecore metamodel (only when
   `format == "ecore"`), reproducing what EMF exposes and pyecore does not:
   - `EClass.eSuperTypes` ← for each `eGenericSuperTypes` child, its
     `eClassifier` target; merged with any direct `eSuperTypes` attribute;
     dedup on (src,dst,type).
   - `ETypedElement.eType` ← `eGenericType.eClassifier` when no direct
     `eType` edge was produced.
   - A reverse edge is written only when the feature records an opposite and
     that opposite feature exists and is not derived (`add_value_edge`).
     On the bundled Ecore metamodel, `EReference.eOpposite` records no
     opposite, so that edge is written only where the file serialises it.
     A symmetric extra pass is not applied: it adds edges the published
     graphs do not contain.
   - Derived/`eStructuralFeature.derived` features: skip edges for features
     marked `derived="true"` in the metamodel (the reference parser skips
     `f.isDerived()`). Notably the reference `eSuperTypes` is NOT marked
     derived in Ecore.ecore even though EMF derives it — produce edges for it
     anyway per the rules above.
7. `attrs` values: raw strings (numbers stay strings — downstream encoders
   cast; keep it lossless). Enum attributes are serialized as strings in XMI
   too — nothing special needed.

## Performance

On the 500 Ecore models, one fresh process of the current loader plus the
NetworkX encoding takes 1.11±0.02 s (stage 1: 0.67±0.01 s, stage 2:
0.45±0.01 s; ten runs, `experiments/results/timing_paired_ecore_final.json`).
An earlier PyEcore loader took 7.98±0.11 s and is not the timed tool.
The loader parses each file once, resolves in one pass, and does not recurse
into external resources.

## What was checked

The edge-preserving certificate against the graphs published by López et al.
is 500/500 Ecore, 500/500 RDS and 369/369 Yakindu
(`experiments/results/fidelity_cert.json`). `moegl/` does not import pyecore.

Note: `metamodels/ecore/Ecore.ecore`, `metamodels/rds/rds_manual.ecore`,
`metamodels/yakindu/*.ecore` are already copied in the repo. `input.format:
"xmi"` + `input.metamodelpath` may be a file or a directory of ecores.
