# nlp_task2vec/fim_text.py
import torch, collections

@torch.no_grad()
def layer_names(module):
    # 依參數名推回 layer 粗粒度：embeddings / encoder.layer.N / others
    names = collections.OrderedDict()
    for n,_ in module.named_parameters():
        if not _.requires_grad: continue
        # 不會發生；這裡留著做保險
        pass
    # 真正彙總時，我們看 "backbone" 的參gj4
    return

def fim_diag_by_layer(model, backbone_attr="backbone"):
    """
    回傳一個字典 {layer_name: fisher_value}
    """
    # 先建立對 backbone 參數名 -> 層名 的對映
    layer_map = {}
    for n,p in getattr(model, backbone_attr).named_parameters():
        if not p.requires_grad:  # backbone 是凍結；但我們仍要抓梯度（需手動啟用）
            p.requires_grad_(True)
        if n.startswith("embeddings."):
            layer = "embeddings"
        elif n.startswith("encoder.layer."):
            # encoder.layer.0.xx -> layer_00
            idx = n.split(".")[2]
            layer = f"layer_{int(idx):02d}"
        else:
            layer = "other"
        layer_map[n] = (p, layer)

    # 建立對角 Fisher 暫存
    fisher_accum = {layer: 0.0 for layer in set([v[1] for v in layer_map.values()])}
    counts = {k: 0 for k in fisher_accum}

    def accumulate():
        for n,(p,layer) in layer_map.items():
            if p.grad is None: continue
            g2 = (p.grad.detach() ** 2).sum().item()
            fisher_accum[layer] += g2
            counts[layer] += 1

    return fisher_accum, counts, layer_map

def normalize_layerwise(fisher_accum, counts, eps=1e-12):
    out = []
    keys = sorted(fisher_accum.keys())
    for k in keys:
        v = fisher_accum[k] / max(counts[k], 1)
        out.append(v)
    import numpy as np
    v = np.array(out, dtype="float64")
    # log + L2 normalize（穩定一點）
    v = np.log1p(v)
    v = v / (np.linalg.norm(v) + eps)
    return keys, v