"""Resolve the Yakindu pairs whose signature matched but greedy refinement did not."""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

from fidelity_cert import signature  # noqa: E402
from moegl import build_graphs, extract, load_config  # noqa: E402
from run_experiments import CONFIGS, DATASETS, published_graph, refinement_mapping  # noqa: E402


def typed_iso(a, b):
    return nx.is_isomorphic(
        a, b,
        node_match=lambda x, y: x.get("type") == y.get("type"),
        edge_match=lambda x, y: x.get("type") == y.get("type"),
    )


def main():
    ds = DATASETS["yakindu"]
    cfg = load_config(os.path.join(CONFIGS, ds["config"]))
    graphs = build_graphs(extract(cfg), cfg)
    buckets = defaultdict(list)
    for name, g in graphs.items():
        buckets[signature(g)].append(name)
    # Rebuild both sides of every signature where greedy refinement missed.
    pub_by_sig = defaultdict(list)
    for pf in sorted(f for f in os.listdir(ds["published"]) if f.endswith(".json")):
        g = published_graph(os.path.join(ds["published"], pf))
        pub_by_sig[signature(g)].append((pf, g))
    moegl_by_sig = defaultdict(list)
    for name, g in graphs.items():
        moegl_by_sig[signature(g)].append((name, g))
    hard_sigs = []
    for sig, pubs in pub_by_sig.items():
        mo = moegl_by_sig.get(sig, [])
        if len(mo) < len(pubs):
            continue
        # greedy: if every published graph gets a refinement hit, skip
        used = set()
        missed = False
        for _, g in pubs:
            hit = None
            for i, (name, mg) in enumerate(mo):
                if i in used:
                    continue
                if refinement_mapping(mg, g) is not None:
                    hit = i
                    break
            if hit is None:
                missed = True
                break
            used.add(hit)
        if missed:
            hard_sigs.append(sig)
    print(f"signatures where greedy refinement is incomplete: {len(hard_sigs)}", flush=True)
    rescued = still = 0
    for sig in hard_sigs:
        pubs = pub_by_sig[sig]
        mo = moegl_by_sig[sig]
        m = len(pubs)
        adj = [[typed_iso(mo[j][1], pubs[i][1]) for j in range(len(mo))] for i in range(m)]
        # maximum matching, published -> one moegl graph
        best = 0

        def search(i, used):
            nonlocal best
            if i == m:
                best = max(best, m)
                return True
            for j in range(len(mo)):
                if j in used or not adj[i][j]:
                    continue
                used.add(j)
                if search(i + 1, used):
                    return True
                used.remove(j)
            return False

        search(0, set())
        # best stays 0 if no complete matching; count how many can match
        if best == m:
            rescued += m
            print(f"  signature size {m}: all {m} isomorphic under a better pairing", flush=True)
        else:
            # count maximum cardinality instead
            best_n = 0

            def card(i, used, acc):
                nonlocal best_n
                if acc + (m - i) <= best_n:
                    return
                if i == m:
                    best_n = max(best_n, acc)
                    return
                card(i + 1, used, acc)
                for j in range(len(mo)):
                    if j in used or not adj[i][j]:
                        continue
                    used.add(j)
                    card(i + 1, used, acc + 1)
                    used.remove(j)

            card(0, set(), 0)
            rescued += best_n
            still += m - best_n
            print(
                f"  signature size {m} moegl {len(mo)} nodes {pubs[0][1].number_of_nodes()}: "
                f"matched {best_n}, differ {m - best_n}",
                flush=True,
            )
    print(f"rescued {rescued} still_different {still}", flush=True)


if __name__ == "__main__":
    main()
