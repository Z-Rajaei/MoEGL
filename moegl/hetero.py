"""Stage 2b (optional): adapted ModelCache -> torch_geometric HeteroData.

torch / torch_geometric are imported lazily so the core package stays free of
them. Node feature tensors concatenate the encoded attributes of the node's
type (sorted by attribute key); types without attributes get ``torch.ones``.
"""

from .adapt import adapt
from .encoders import AttributeEncoder


def build_hetero(cache, config):
    """Return {model_name: HeteroData}."""
    import torch
    from torch_geometric.data import HeteroData

    adapted = adapt(cache, config)
    encoder = AttributeEncoder(
        adapted["meta"].get("encodings", {}),
        w2v_dim=config.w2v_dim,
        seed=config.seed,
    ).fit(adapted["models"])

    # attribute layout per node type is fixed over the whole dataset, so the
    # feature tensors of one type have the same width in every graph
    widths = {}  # node type -> {attribute key: width}
    for model in adapted["models"]:
        for node in model["nodes"]:
            tw = widths.setdefault(node["type"], {})
            for k, v in node["attrs"].items():
                if k not in tw:
                    tw[k] = len(encoder.encode_numeric(node["type"], k, v))

    datas = {}
    for model in adapted["models"]:
        data = HeteroData()

        # group nodes by type, preserving extraction order inside each type
        by_type = {}
        for node in model["nodes"]:
            by_type.setdefault(node["type"], []).append(node)

        local_id = {}  # global node id -> (type, per-type index)
        for ntype, nodes in by_type.items():
            attr_keys = sorted(widths[ntype])
            rows, orig_ids = [], []
            for i, node in enumerate(nodes):
                local_id[node["id"]] = (ntype, i)
                orig_ids.append(node["id"])
                if attr_keys:
                    vec = []
                    for k in attr_keys:
                        if k in node["attrs"]:
                            vec.extend(
                                encoder.encode_numeric(ntype, k, node["attrs"][k])
                            )
                        else:
                            vec.extend([0.0] * widths[ntype][k])
                    rows.append(vec)
                else:
                    rows.append([1.0])
            data[ntype].x = torch.tensor(rows, dtype=torch.float)
            data[ntype].node_id = torch.arange(len(nodes), dtype=torch.long)
            data[ntype].orig_id = torch.tensor(orig_ids, dtype=torch.long)

        # group edges by (src_type, edge_type, dst_type)
        edge_groups = {}
        for edge in model["edges"]:
            s = local_id.get(edge["src"])
            d = local_id.get(edge["dst"])
            if s is None or d is None:
                continue
            key = (s[0], edge["type"], d[0])
            edge_groups.setdefault(key, []).append((s[1], d[1]))
        for key, pairs in edge_groups.items():
            ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            data[key].edge_index = ei

        datas[model["name"]] = data
    return datas
