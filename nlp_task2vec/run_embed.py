# nlp_task2vec/run_embed.py
import os, argparse, json, math, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from nlp_task2vec.data_cl_benchmark import dataset_roots, load_split
from nlp_task2vec.modeling_probe import ProbeClassifier, get_tokenizer
from nlp_task2vec.fim_text import fim_diag_by_layer, normalize_layerwise

class TxtClsDS(Dataset):
    def __init__(self, df, tok, max_len=256):
        self.sentence = df["sentence"].tolist()
        self.label = df["label"].tolist()
        self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.sentence)
    def __getitem__(self, i):
        enc = self.tok(self.sentence[i], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        item = {k:v.squeeze(0) for k,v in enc.items()}
        item["labels"] = torch.tensor(self.label[i], dtype=torch.long)
        return item

def train_head(model, dl, epochs=2, lr=1e-3, device="cuda", use_amp=True):
    model.to(device)
    for p in model.backbone.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    total_steps = epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)
    model.train()
    for ep in range(epochs):
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids, attn)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
    return model

@torch.no_grad()
def accuracy(model, dl, device="cuda", id2label=None):
    model.eval()
    n=0
    c=0
    batch_counter = 0

    for b in dl:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        predictions = logits.argmax(-1).cpu().tolist()
        labels = b["labels"].tolist()
        # 僅印出前幾個批次的結果
        if batch_counter < 1:
            print(f"[{'='*5}] Batch {batch_counter} 預測 vs. 實際標籤:")
            for i, (pred, true) in enumerate(zip(predictions, labels)):
                # 將數字標籤轉換回文字標籤
                pred_label = id2label[pred] if id2label else pred
                true_label = id2label[true] if id2label else true
                print(f"  Sample {i}: PRED='{pred_label}' | GOLD='{true_label}'")
            batch_counter += 1
        c += (logits.argmax(-1).cpu()==b["labels"]).sum().item()
        n += len(b["labels"])
    return c/max(n,1)

def mc_fisher(model, dl, device="cuda", mc_steps=1):
    model.eval()
    fisher_accum, counts, layer_map = fim_diag_by_layer(model)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids, attn)
            probs = torch.softmax(logits, dim=-1)

        for _ in range(mc_steps):
            # 從模型預測分佈中抽樣 y
            y = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # 重新 forward（允許梯度）
            for p in model.parameters():
                if p.grad is not None: p.grad = None
            logits2 = model(input_ids, attn)
            loss = nn.functional.cross_entropy(logits2, y, reduction="sum")
            loss.backward()
            # 彙總本 batch 的梯度平方（對角 Fisher 近似）
            for n,(p,layer) in layer_map.items():
                if p.grad is None: continue
                # 累積 sum(g^2)；稍後再做平均與 normalize
                counts[layer] += 1
            # 清空梯度以免疊加到下一次
            for p in model.parameters():
                if p.grad is not None:
                    g2 = (p.grad.detach()**2).sum().item()
                    # 直接把所有梯度都加入同一 layer 的 total
                    # （注意：這裡為了效率，只取 sum，不逐參數記）
                    # 由於每層參數量不同，做 log 與 L2 normalize
                    layer = None
            # 真正逐參數歸屬已在 fim_diag_by_layer 做了對映，
            # 但為了避免過多 Python 回圈，這裡簡化為：用 p.name 的上層 layer 直接加總
            # 改用一次性迴圈
            for n,(p,layer) in layer_map.items():
                if p.grad is not None:
                    g2 = (p.grad.detach()**2).sum().item()
                    fisher_accum[layer] += g2

    keys, vec = normalize_layerwise(fisher_accum, counts)
    return keys, vec

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)

    roots = dataset_roots(args.data_root)
    backbone = args.backbone

    all_embeds = []
    all_names = []

    for name in args.datasets.split(","):
        name = name.strip()
        print(f"\n=== Processing dataset: {name} ===")   # 顯示 dataset 名稱
        data_dir = roots[name]
        tr, dv, te, id2label = load_split(data_dir)
        tok = get_tokenizer(backbone)
        # 建 train dataloader（小樣本即可）
        train_ds = TxtClsDS(tr.sample(min(len(tr), args.max_train)), tok, args.max_len)
        dev_ds   = TxtClsDS(dv if dv is not None else tr.sample(min(len(tr), 2000)), tok, args.max_len)
        te_ds    = TxtClsDS(te if te is not None else dev_ds, tok, args.max_len)
        num_labels = max(tr["label"].max(), (dv["label"].max() if dv is not None else 0), (te["label"].max() if te is not None else 0)) + 1

        model = ProbeClassifier(backbone, num_labels=num_labels)
        train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4)
        dev_dl   = DataLoader(dev_ds, batch_size=args.bs, shuffle=False, num_workers=4)

        # 訓練分類頭 1~2 個 epoch
        print(f"[{name}] Training classification head...")
        model = train_head(model, train_dl, epochs=args.epochs, lr=args.lr, device=device, use_amp=not args.no_amp)
        acc = accuracy(model, dev_dl, device=device, id2label=id2label)
        print(f"[{name}] head-only dev accuracy: {acc:.4f}")

        # 用 dev set 估計 Fisher
        print(f"[{name}] Estimating Fisher embedding...")
        keys, vec = mc_fisher(model, dev_dl, device=device, mc_steps=args.mc_steps)
        all_embeds.append(vec); all_names.append(name)

        # 存單一任務向量
        np.save(os.path.join(args.out_dir, f"{name}_embed.npy"), vec)
        with open(os.path.join(args.out_dir, f"{name}_layers.json"), "w") as f:
            json.dump({"layers": keys}, f, indent=2)
        print(f"[{name}] Fisher embedding saved.\n")  # 顯示保存完成

    # 距離矩陣（1 - cosine）
    print("Calculating distance matrix...")
    from sklearn.metrics.pairwise import cosine_similarity
    M = cosine_similarity(np.stack(all_embeds, axis=0))
    D = 1 - M
    df = pd.DataFrame(D, index=all_names, columns=all_names)
    df.to_csv(os.path.join(args.out_dir, "distance_matrix.csv"))
    print("Done. Saved to", args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/CL_Benchmark", type=str)
    ap.add_argument("--datasets", default="dbpedia,amazon,yahoo,agnews", type=str)
    ap.add_argument("--backbone", default="roberta-base", type=str)
    ap.add_argument("--max_len", default=256, type=int)
    ap.add_argument("--max_train", default=20000, type=int, help="最多取多少 train 樣本訓練分類頭（速度考量）")
    ap.add_argument("--epochs", default=5, type=int)
    ap.add_argument("--bs", default=32, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--mc_steps", default=2, type=int)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out_dir", default="outputs_nlp", type=str)
    args = ap.parse_args()
    main(args)