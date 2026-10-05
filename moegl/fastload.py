"""Fast XMI/Ecore model loading with lxml -- replaces pyecore (docs/loader.md).

Two public functions:

* :func:`metamodel_index` -- parse one or more ``.ecore`` metamodel files into a
  plain-dict index ``{nsURI: {"prefix": str, "classes": {name: class}} }``.
* :func:`parse_model` -- parse one model file in a single lxml pass into the
  ``{"nodes", "edges"}`` shape expected by the ModelCache (plus the list of
  self-reference repairs that were applied).

Semantics intentionally reproduce the observable behaviour of EMF's
``XMIResourceImpl`` + ``getAllContents`` (as previously approximated through
pyecore): document-order DFS node ids, same-resource edges only, proxies and
unresolvable fragments dropped, containment opposite wiring, and the EMF-only
``eSuperTypes`` / ``eType`` / ``eOpposite`` derivations for ``format="ecore"``.
"""

import glob
import os
import re

from lxml import etree

XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
XMI_NS = "http://www.omg.org/XMI"
ECORE_NS = "http://www.eclipse.org/emf/2002/Ecore"

_XSI_TYPE = "{%s}type" % XSI_NS
_XSI_NIL = "{%s}nil" % XSI_NS
_XMI_ID = "{%s}id" % XMI_NS
_XMI_TYPE = "{%s}type" % XMI_NS

_PARSER = etree.XMLParser(remove_comments=True, recover=False)
_ROOTNUM = re.compile(r"^/(\d+)")

# metamodels/ecore/Ecore.ecore relative to the moegl package
BUNDLED_ECORE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metamodels", "ecore", "Ecore.ecore",
)


class LoadError(Exception):
    """A model file cannot be decoded against the metamodel index."""


# --------------------------------------------------------------------------- index
def _last_name(ref):
    """Last named fragment segment of an EMF reference.

    ``"ecore:EDataType http://...#//EString" -> "EString"``,
    ``"#//A/B" -> "B"``, ``"#/1/X" -> "X"``, ``"#//@eClassifiers.3" ->
    "eClassifiers"``.
    """
    frag = ref.rsplit("#", 1)[-1]
    if " " in frag:
        frag = frag.split()[-1]
    seg = frag.rsplit("/", 1)[-1]
    if seg.startswith("@"):
        seg = seg[1:].split(".", 1)[0]
    return seg


def _index_feature(elem):
    """One ``eStructuralFeatures`` element of a metamodel -> feature entry."""
    name = elem.get("name")
    if not name:
        return None
    xtype = elem.get(_XSI_TYPE) or elem.get(_XMI_TYPE) or ""
    tname = xtype.rpartition(":")[2]
    if tname == "EAttribute":
        kind = "attr"
    elif tname == "EReference":
        kind = "ref"
    else:  # no xsi:type: declared type is the abstract EStructuralFeature
        kind = "ref" if elem.get("eOpposite") else "attr"
    upper = elem.get("upperBound")
    etype = elem.get("eType")
    opposite = elem.get("eOpposite")
    return {
        "name": name,
        "kind": kind,
        "type": _last_name(etype) if etype else None,
        "many": upper not in (None, "1"),
        "containment": elem.get("containment") == "true",
        "derived": elem.get("derived") == "true",
        "iD": elem.get("iD") == "true",
        "opposite": _last_name(opposite) if opposite else None,
    }


def _index_classifier(elem, index, ns_uri, ns_prefix):
    pkg = index.setdefault(ns_uri, {"prefix": ns_prefix, "classes": {}})
    if ns_prefix and not pkg["prefix"]:
        pkg["prefix"] = ns_prefix
    name = elem.get("name")
    if not name or name in pkg["classes"]:
        return  # first definition wins (nested packages may share a nsURI)
    xtype = elem.get(_XSI_TYPE) or elem.get(_XMI_TYPE) or "ecore:EClass"
    supers, seen = [], set()
    for tok in (elem.get("eSuperTypes") or "").split():
        n = _last_name(tok)
        if n and n not in seen:
            seen.add(n)
            supers.append(n)
    features = {}
    for child in elem:
        if not isinstance(child.tag, str):
            continue
        local = etree.QName(child).localname
        if local == "eStructuralFeatures":
            f = _index_feature(child)
            if f is not None:
                features[f["name"]] = f
        elif local == "eGenericSuperTypes":
            c = child.get("eClassifier")
            if c:
                n = _last_name(c)
                if n and n not in seen:
                    seen.add(n)
                    supers.append(n)
    pkg["classes"][name] = {
        "name": name,
        "kind": xtype.rpartition(":")[2],
        "abstract": elem.get("abstract") == "true",
        "supertypes": supers,
        "features": features,
        "_ns": ns_uri,
    }


def _index_ecore_file(path, index):
    tree = etree.parse(path, _PARSER)
    stack = [(tree.getroot(), "", "")]
    while stack:
        elem, ns_uri, ns_prefix = stack.pop()
        if not isinstance(elem.tag, str):
            continue
        if "nsURI" in elem.attrib:
            ns_uri = elem.get("nsURI") or ""
            ns_prefix = elem.get("nsPrefix") or ""
        if etree.QName(elem).localname == "eClassifiers":
            _index_classifier(elem, index, ns_uri, ns_prefix)
        stack.extend((c, ns_uri, ns_prefix) for c in elem)


def metamodel_index(path_or_dir):
    """Build the metamodel index from a .ecore file or a directory of them."""
    index = {}
    if not path_or_dir:
        return index
    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, "*.ecore")))
    elif os.path.isfile(path_or_dir):
        files = [path_or_dir]
    else:
        return index
    for f in files:
        _index_ecore_file(f, index)
    return index


def _class_by_name(index, name, ns_uri=None):
    if ns_uri is not None:
        pkg = index.get(ns_uri)
        if pkg is not None and name in pkg["classes"]:
            return pkg["classes"][name]
    for pkg in index.values():
        entry = pkg["classes"].get(name)
        if entry is not None:
            return entry
    return None


def _all_features(index, entry):
    """own features + supertypes' features, recursively (memoised on the entry)."""
    feats = entry.get("_allf")
    if feats is None:
        feats = dict(entry["features"])
        entry["_allf"] = feats  # placed before recursion: breaks supertype cycles
        for sup in entry["supertypes"]:
            se = _class_by_name(index, sup, entry["_ns"])
            if se is None:
                continue
            for k, v in _all_features(index, se).items():
                feats.setdefault(k, v)
    return feats


def transitive_supertypes(index):
    """Return a function class_name -> sorted list of transitive supertype names."""
    memo = {}

    def supers_of(entry):
        out, stack = set(), [entry]
        while stack:
            e = stack.pop()
            cached = memo.get(id(e))
            if cached is not None:
                out.update(cached)
                continue
            for s in e["supertypes"]:
                if s in out:
                    continue
                out.add(s)
                se = _class_by_name(index, s, e["_ns"])
                if se is not None:
                    stack.append(se)
        memo[id(entry)] = out
        return out

    def for_name(name):
        entry = _class_by_name(index, name)
        return sorted(supers_of(entry)) if entry is not None else []

    return for_name


# --------------------------------------------------------------------------- model parsing
def parse_model(path, index, fmt):
    """Parse one XMI/Ecore model file.

    Returns ``(model_dict, repaired)`` where ``model_dict`` is
    ``{"nodes": [...], "edges": [...]}`` and ``repaired`` is a list of
    ``[filename, description]`` entries for self-references that were made
    local. Raises :class:`LoadError` (or ``etree`` errors) on undecodable
    files -- the caller records them in ``meta["skipped"]``.
    """
    tree = etree.parse(path, _PARSER)
    root = tree.getroot()
    nsmap = dict(root.nsmap)
    xmi_ns = nsmap.get("xmi", XMI_NS)
    if root.tag == "{%s}XMI" % xmi_ns:
        xml_roots = [c for c in root if isinstance(c.tag, str)]
    else:
        xml_roots = [root]
    if not xml_roots:
        raise LoadError("model file has no root object")
    # every root unresolvable -> caller still sees an empty model, not a crash

    objs, roots, uuid = [], [], {}
    edges, edgeset = [], set()
    repaired_uris = set()

    def class_of(ns_uri, local, tolerant=False):
        pkg = index.get(ns_uri)
        entry = pkg["classes"].get(local) if pkg is not None else None
        if entry is None and not tolerant:
            raise LoadError(
                "class %r not found in metamodel %r" % (local, ns_uri))
        return entry

    def new_obj(entry, frag):
        o = {
            "id": len(objs),
            "type": entry["name"],
            "attrs": {},
            "_e": entry,      # metamodel class entry
            "_ch": [],        # [(feature name, child obj)] in document order
            "_pd": [],        # [(feature name, feature entry, raw value)]
            "_rf": {},        # feature name -> [resolved same-resource targets]
        }
        objs.append(o)
        return o

    def add_edge(src, dst, fname):
        key = (src["id"], dst["id"], fname)
        if key not in edgeset:
            edgeset.add(key)
            edges.append({"src": src["id"], "dst": dst["id"], "type": fname})

    def add_value_edge(src, dst, f):
        """Edge for one feature value + EMF's eOpposite wiring."""
        if not f["derived"]:
            add_edge(src, dst, f["name"])
        opp = f["opposite"]
        if opp:
            of = _all_features(index, dst["_e"]).get(opp)
            if of is not None and not of["derived"]:
                add_edge(dst, src, opp)

    # -- fragment / reference resolution -------------------------------------
    def named_child(obj, key):
        # pyecore navigation order: eSubpackages, then eClassifiers, then any
        # named child. Built once per object; the first match in that order wins.
        idx = obj.get("_by")
        if idx is None:
            idx = {}
            for fname in ("eSubpackages", "eClassifiers"):
                for f, c in obj["_ch"]:
                    if f != fname:
                        continue
                    n = c["attrs"].get("name")
                    if n not in idx:
                        idx[n] = c
            for f, c in obj["_ch"]:
                n = c["attrs"].get("name")
                if n not in idx:
                    idx[n] = c
            obj["_by"] = idx
        return idx.get(key)

    def resolve(frag):
        """Resolve an intra-resource URI fragment to an object or None."""
        if " " in frag:
            frag = frag.split()[-1]
        if not frag:
            return None
        if not frag.startswith("/"):
            return uuid.get(frag)  # xmi:id / iD attribute value
        m = _ROOTNUM.match(frag)
        rootnum = int(m.group(1)) if m else 0
        rest = frag[m.end():] if m else frag
        if rootnum >= len(roots):
            return None
        obj = roots[rootnum]
        annot = False
        for seg in rest.split("/"):
            if not seg:
                continue
            if annot:
                annot = False
                obj = named_child(obj, seg)
            elif seg.startswith("@"):
                fname, _, idx = seg[1:].partition(".")
                kids = [c for f, c in obj["_ch"] if f == fname]
                try:
                    obj = kids[int(idx)] if idx else kids[0]
                except (IndexError, ValueError):
                    return None
            elif seg.startswith("%"):
                key = seg[1:-1] if seg.endswith("%") else seg[1:]
                obj = next(
                    (c for f, c in obj["_ch"]
                     if c["type"] == "EAnnotation"
                     and c["attrs"].get("source") == key),
                    None,
                )
                annot = obj is not None
            else:
                obj = named_child(obj, seg)
            if obj is None:
                return None
        return obj

    def resolve_external(uri, frag):
        """Cross-resource URI part of a reference token."""
        if ":/" not in uri and os.path.basename(uri) == os.path.basename(path):
            return resolve(frag)  # reference under this file's own name: internal
        if ":/" in uri or uri.startswith(("platform:", "pathmap:")):
            return None  # absolute/scheme URI: out of scope
        if os.path.exists(os.path.join(os.path.dirname(path), uri)):
            return None  # real external resource: out of scope (published graphs
            # only contain same-resource nodes)
        obj = resolve(frag)
        if obj is not None:
            # resource was renamed after the model was written: the fragment
            # resolves locally, so treat it as a self-reference (EMF repair)
            repaired_uris.add(uri)
        return obj

    def resolve_token(token):
        if "#" in token:
            uri, frag = token.rsplit("#", 1)
            if uri:
                return resolve_external(uri, frag)
            return resolve(frag)
        if ":" in token and "/" not in token and token.split(":", 1)[0] in nsmap:
            return None  # 'prefix:Type' qualifier of a typed reference
        return resolve(token)

    # -- feature decoding -----------------------------------------------------
    def elem_entry(elem, f, owner_entry):
        """Concrete class of an object-valued child element."""
        xtype = elem.get(_XSI_TYPE) or elem.get(_XMI_TYPE)
        if xtype:
            prefix, sep, local = xtype.partition(":")
            if not sep:
                raise LoadError("malformed xsi:type %r" % xtype)
            ns = elem.nsmap.get(prefix)
            if ns is None:
                return None  # foreign xsi:type prefix: skip the subtree
            return class_of(ns, local, tolerant=True)
        if f["type"]:
            return _class_by_name(index, f["type"], owner_entry["_ns"])
        return None  # untyped element: skip the subtree

    def decode_attrs(o, elem, entry, is_root):
        for key, val in elem.attrib.items():
            if key == _XMI_ID:
                uuid[val] = o
                continue
            if key in (_XSI_TYPE, _XMI_TYPE, _XSI_NIL) or key.startswith("{"):
                continue  # xsi:/xmi:/other-namespaced attributes are not features
            if key == "href":
                continue
            f = _all_features(index, entry).get(key)
            if f is None:
                continue  # unknown attribute: EMF ignores foreign features
            if f["kind"] == "attr":
                if f["derived"]:
                    continue
                o["attrs"][key] = val.split() if f["many"] else val
                if f["iD"]:
                    uuid[val] = o
            else:
                o["_pd"].append((f, val))

    # -- tree walk: DFS over containment children, one root after the other --
    # (document order == EMF getAllContents order for these serialisations)
    for i, xel in enumerate(xml_roots):
        q = etree.QName(xel)
        entry = class_of(q.namespace or "", q.localname, tolerant=True)
        if entry is None:
            continue  # root from a foreign metamodel (e.g. GMF notation): skip
        o = new_obj(entry, "/%d" % i)
        roots.append(o)
        decode_attrs(o, xel, entry, True)
        frag = "/%d" % i
        stack = [(o, frag, iter(xel))]
        while stack:
            o, frag, it = stack[-1]
            try:
                child = next(it)
            except StopIteration:
                stack.pop()
                continue
            if not isinstance(child.tag, str):
                continue  # processing instructions etc.
            tag = etree.QName(child).localname
            f = _all_features(index, o["_e"]).get(tag)
            if f is None:
                continue  # foreign element (e.g. GMF notation): skip subtree
            href = child.get("href")
            if href is not None:
                o["_pd"].append((f, href))
                continue
            if f["derived"]:
                continue  # objects under derived features are unreachable
            if child.get(_XSI_NIL) is not None:
                if f["kind"] == "attr":
                    if f["many"]:
                        o["attrs"].setdefault(tag, []).append(None)
                    else:
                        o["attrs"][tag] = None
                continue
            if f["kind"] == "attr":
                text = child.text or ""
                if f["many"]:
                    o["attrs"].setdefault(tag, []).append(text)
                else:
                    o["attrs"][tag] = text
                continue
            centry = elem_entry(child, f, o["_e"])
            if centry is None:
                continue  # unresolvable type: skip the subtree
            if centry["kind"] == "EDataType":
                # element whose type is a datatype decodes as an attribute value
                text = child.text or ""
                if f["many"]:
                    o["attrs"].setdefault(tag, []).append(text)
                else:
                    o["attrs"][tag] = text
                continue
            if centry["name"] == "EStringToStringMapEntry" and tag == "details":
                continue  # EAnnotation details decode into a map: not a node
            if not f["containment"]:
                continue  # non-containment element: unreachable orphan
            counts = o.setdefault("_tc", {})
            j = counts.get(tag, 0)
            counts[tag] = j + 1
            cfrag = "%s/@%s.%d" % (frag, tag, j)
            cobj = new_obj(centry, cfrag)
            o["_ch"].append((tag, cobj))
            decode_attrs(cobj, child, centry, False)
            add_value_edge(o, cobj, f)
            stack.append((cobj, cfrag, iter(child)))

    # -- resolve deferred references -------------------------------------------
    for o in objs:
        for f, raw in o["_pd"]:
            if f["derived"]:
                continue
            for token in raw.split():
                dst = resolve_token(token)
                if dst is not None:
                    lst = o["_rf"].setdefault(f["name"], [])
                    if dst not in lst:
                        lst.append(dst)
                    add_value_edge(o, dst, f)
                else:
                    # External reference. EMF writes it as two tokens,
                    # "ecore:EDataType" plus "http://...#//EString". Only the
                    # fragment after '#' is the datatype name; the qualifier
                    # is not a model element.
                    if "#" not in token:
                        continue
                    name = _last_name(token)
                    if name:
                        o["attrs"][f["name"] + "Name"] = name

    # -- EMF derivations specific to the Ecore metamodel ----------------------
    if fmt == "ecore":
        has_etype = {e["src"] for e in edges if e["type"] == "eType"}

        def erasure(g):
            """Erasure of an EGenericType object -> EClassifier node or None."""
            seen = set()
            while g is not None and id(g) not in seen:
                seen.add(id(g))
                cl = g["_rf"].get("eClassifier")
                if cl:
                    return cl[0]
                tps = g["_rf"].get("eTypeParameter")
                if not tps:
                    return None
                # bound erasure: first eBounds of the ETypeParameter
                g = next((c for f, c in tps[0]["_ch"] if f == "eBounds"), None)
            return None

        for o in objs:
            # EClass.eSuperTypes <- eGenericSuperTypes.eClassifier
            for fname, g in o["_ch"]:
                if fname == "eGenericSuperTypes":
                    t = erasure(g)
                    if t is not None:
                        add_edge(o, t, "eSuperTypes")
            # ETypedElement.eType <- eGenericType erasure (if no direct eType)
            if o["id"] not in has_etype:
                for fname, g in o["_ch"]:
                    if fname == "eGenericType":
                        t = erasure(g)
                        if t is not None:
                            add_edge(o, t, "eType")
                        elif "eTypeName" not in o["attrs"] and g["attrs"].get("eClassifierName"):
                            o["attrs"]["eTypeName"] = g["attrs"]["eClassifierName"]
                        break
        # NOTE: EReference.eOpposite is NOT auto-wired in the other direction:
        # the published graphs contain exactly the serialized eOpposite edges
        # (validated empirically: the symmetric rule produces spurious edges).

    nodes = [{"id": o["id"], "type": o["type"], "attrs": o["attrs"]} for o in objs]
    repaired = [
        [os.path.basename(path), "self-references to %s made local" % u]
        for u in sorted(repaired_uris)
    ]
    return {"nodes": nodes, "edges": edges}, repaired
