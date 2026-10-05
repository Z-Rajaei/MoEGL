"""Attribute value encoders: raw, one-hot, word2vec, worde4mde.

Vocabularies and the word2vec model are fitted once over the whole dataset
(``fit``), then applied per value (``encode``). Adapted from the AutoMol
feature encoder design (tokenize / one-hot / dataset-wide word2vec fit).
Encodings are keyed by (node type, attribute key) as produced by ``adapt``.
"""

import re

import numpy as np

_CAMEL_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")


def tokenize(value):
    """Split an identifier into lowercase tokens (camelCase/snake_case/digits)."""
    s = _NON_ALNUM.sub(" ", str(value))
    s = _CAMEL_1.sub(r"\1 \2", s)
    s = _CAMEL_2.sub(r"\1 \2", s)
    return [t for t in s.lower().split() if t]


def to_float(value):
    s = str(value).strip().lower()
    if s == "true":
        return 1.0
    if s == "false":
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _one_hot(value, vocab):
    vec = [0.0] * (len(vocab) + 1)
    try:
        vec[vocab.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


class AttributeEncoder:
    """Fits and applies the encodings declared in the DSL configuration."""

    def __init__(self, encodings, w2v_dim=64, seed=42):
        """encodings: {(node type, attribute key) -> encoding}"""
        self.encodings = dict(encodings or {})
        self.w2v_dim = w2v_dim
        self.seed = seed
        self.onehot_vocabs = {}
        self.w2v = None
        self._w2v_keys = set()
        self._e4mde = {}
        self._fitted = False

    def fit(self, models):
        onehot_values = {
            k: set() for k, e in self.encodings.items() if e == "one-hot"
        }
        self._w2v_keys = {
            k for k, e in self.encodings.items() if e == "word2vec"
        }
        e4mde_keys = {k for k, e in self.encodings.items() if e == "worde4mde"}
        sentences = []

        for model in models:
            for node in model.get("nodes", []):
                attrs = node.get("attrs", {})
                for key in onehot_values:
                    if key[0] == node["type"] and key[1] in attrs:
                        onehot_values[key].add(str(attrs[key[1]]))
                for key in self._w2v_keys:
                    if key[0] == node["type"] and key[1] in attrs:
                        sentences.append(tokenize(attrs[key[1]]))

        self.onehot_vocabs = {k: sorted(v) for k, v in onehot_values.items()}

        if sentences:
            from gensim.models import Word2Vec

            self.w2v = Word2Vec(
                sentences=sentences,
                vector_size=self.w2v_dim,
                min_count=1,
                window=3,
                sg=1,
                seed=self.seed,
                workers=1,
                epochs=20,
            )

        if e4mde_keys:
            try:
                from worde4mde import load_embeddings
            except ImportError as exc:
                raise ImportError(
                    "encoding 'worde4mde' requires the worde4mde package "
                    "(pip install worde4mde)"
                ) from exc
            kv = load_embeddings("sgram-mde")
            self._e4mde = {k: kv for k in e4mde_keys}

        self._fitted = True
        return self

    def encode(self, node_type, feature, value):
        """Encode one attribute value. Returns the raw value if no encoding is
        declared, a float list for one-hot/word2vec/worde4mde."""
        enc = self.encodings.get((node_type, feature))
        if enc is None or enc == "rawstring":
            return value
        if enc == "one-hot":
            return _one_hot(
                str(value), self.onehot_vocabs.get((node_type, feature), [])
            )
        if enc == "word2vec":
            tokens = tokenize(value)
            if self.w2v is None or not tokens:
                return [0.0] * self.w2v_dim
            vecs = [self.w2v.wv[t] for t in tokens if t in self.w2v.wv]
            if not vecs:
                return [0.0] * self.w2v_dim
            return np.mean(vecs, axis=0).astype(np.float32).tolist()
        if enc == "worde4mde":
            emb = self._e4mde.get((node_type, feature))
            if emb is None:
                return [0.0] * self.w2v_dim
            vecs = [emb[t] for t in tokenize(value) if t in emb.key_to_index]
            if not vecs:
                return [0.0] * emb.vector_size
            return np.mean(vecs, axis=0).astype(np.float32).tolist()
        return value

    def encode_numeric(self, node_type, feature, value):
        """Like encode(), but always returns a flat list of floats (for
        HeteroData tensors). Raw values are coerced with to_float()."""
        enc = self.encodings.get((node_type, feature))
        if enc in (None, "rawstring"):
            if isinstance(value, (list, tuple)):
                return [to_float(v) for v in value]
            return [to_float(value)]
        out = self.encode(node_type, feature, value)
        return (
            list(out)
            if isinstance(out, (list, tuple))
            else [to_float(out)]
        )
