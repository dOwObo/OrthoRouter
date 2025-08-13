# helper/utils.py

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model.layers import MoEBlock

def collate_fn(batch):
    """
    DataLoader 取出的 batch 經常會是一個 list，包含每筆資料的 dict。
    這裡示範將多筆資料組合成可以餵給模型的 tensor。
    """
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = torch.tensor([example["input_ids"] for example in batch], dtype=torch.long).to(target_device)
    attention_mask = torch.tensor([example["attention_mask"] for example in batch], dtype=torch.long).to(target_device)
    labels = torch.tensor([example["labels"] for example in batch], dtype=torch.long).to(target_device)
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

def evaluate(model, dataloader, tokenizer, labels_list):
    device = next(model.parameters()).device
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="evaluation")):
            batch = {k: v.to(device) for k, v in batch.items()}

            # T5 生成
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=10
            )
            # decode predictions
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # decode gold label
            gold_texts = []
            for label_seq in batch["labels"]:
                label_seq = label_seq[label_seq != -100]  # 去掉 -100
                gold_text = tokenizer.decode(label_seq, skip_special_tokens=True).strip()
                gold_texts.append(gold_text)

            # Debug: 印出前幾筆看看預測 vs. 標籤
            if batch_idx < 2:  # 前2個batch做檢查
                for i, (p, g) in enumerate(zip(predictions, gold_texts)):
                    print(f"[DEBUG] batch#{batch_idx} sample#{i}  PRED: '{p}' | GOLD: '{g}'")

            # 比對
            for pred, gold in zip(predictions, gold_texts):
                if pred.strip().lower() == gold.lower():
                    correct += 1
                total += 1

    accuracy = correct / total if total else 0
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

# def visualize_expert_selection(selection_counts, output_dir="."):
#     """
#     繪製各層專家使用頻率。
#     :param selection_counts: dict 或 list，存放各層的專家計數
#     :param output_dir: 輸出檔案路徑
#     """
#     import os
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for layer_idx, layer_selection_counts in enumerate(selection_counts):
#         plt.figure(figsize=(8, 4))
#         plt.bar(range(len(layer_selection_counts)), layer_selection_counts, color='blue')
#         plt.title(f"Layer {layer_idx} Expert Selection Frequency")
#         plt.xlabel("Expert Index")
#         plt.ylabel("Count")
#         output_path = os.path.join(output_dir, f"layer_{layer_idx}_selection.png")
#         plt.savefig(output_path)
#         plt.close()
#         print(f"Layer {layer_idx} figure saved to {output_path}")

# 畫熱力圖用
def visualize_expert_selection(selection_counts, output_dir="."):
    """
    繪製專家使用頻率的熱力圖。
    x 軸為 Experts，y 軸為 Layers，
    每個單元格的數值表示該層中對應專家的選擇次數。
    :param selection_counts: list，存放各層的專家計數
    :param output_dir: 輸出檔案路徑
    """
    # DEL 查看selection_counts
    # print("selection_counts",selection_counts)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 若 selection_counts 不是 numpy 陣列，則嘗試轉換
    if not isinstance(selection_counts, np.ndarray):
        selection_counts = np.array([sc.cpu().numpy() if hasattr(sc, 'cpu') else sc for sc in selection_counts])
    
    # 如果只有一層數據，則調整為 2D (1, n)
    if selection_counts.ndim == 1:
        selection_counts = selection_counts.reshape(1, -1)
    
    # ---- 根據encoder/decoder去切兩張熱力圖 ----
    # 總共有 48 層，0~23 為 Encoder，24~47 為 Decoder 
    selection_counts_encoder = selection_counts[:24]
    selection_counts_decoder = selection_counts[24:48]
    
    # ---- 繪製 Encoder 的熱力圖 ----
    plt.figure(figsize=(10, 6))
    sns.heatmap(selection_counts_encoder, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlabel("Experts")
    plt.ylabel("Layers (Encoder 0~23)")
    plt.title("Expert Selection Heatmap (Encoder)")
    encoder_plot_path = os.path.join(output_dir, "expert_selection_heatmap_encoder.png")
    plt.savefig(encoder_plot_path)
    plt.close()
    print(f"Encoder heatmap saved to {encoder_plot_path}")

    # ---- 繪製 Decoder 的熱力圖 ----
    plt.figure(figsize=(10, 6))
    sns.heatmap(selection_counts_decoder, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlabel("Experts")
    plt.ylabel("Layers (Decoder 24~47)")
    plt.title("Expert Selection Heatmap (Decoder)")
    decoder_plot_path = os.path.join(output_dir, "expert_selection_heatmap_decoder.png")
    plt.savefig(decoder_plot_path)
    plt.close()
    print(f"Decoder heatmap saved to {decoder_plot_path}")

# 模型總參數量
def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

# 激活參數量計算
def count_activated_params_t5moe(model):
    # 先計算非 MoE 部分（所有參數均激活）
    total_params = count_total_params(model)
    all_moe_params = 0
    for module in model.modules():
        if isinstance(module, MoEBlock):
            for expert in module.experts:
                all_moe_params += sum(p.numel() for p in expert.parameters())
    non_moe_params = total_params - all_moe_params

    # 計算 MoE 區塊中，依據路由器選擇被激活的專家參數
    activated_moe_params = 0
    for module in model.modules():
        if isinstance(module, MoEBlock):
            # 假設 top_k = 1，則每個 token 選擇一個專家
            # 取得路由器最後的分數，並計算每個 MoEBlock中被激活的專家索引（對所有 token 取 union）
            top_expert_indices = torch.argmax(module.last_scores, dim=-1)  # shape: (batch_size, seq_length)
            unique_experts = torch.unique(top_expert_indices)
            for idx in unique_experts:
                expert = module.experts[idx]
                activated_moe_params += sum(p.numel() for p in expert.parameters())
    activated_params = non_moe_params + activated_moe_params
    return activated_params