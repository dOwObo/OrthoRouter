import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank: int = 4):
        # ---------------------------------- #
        # 原始 transformer 的 FFN 層加上 LoRA #
        # ---------------------------------- #
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank

        # 固定 LoRA 參數，不進行動態擴展
        self.lora_As = nn.Parameter(torch.zeros((rank, original_layer.wi.weight.size(1))))
        self.lora_Bs = nn.Parameter(torch.zeros((original_layer.wi.weight.size(0), rank)))
       
        # 初始化
        nn.init.kaiming_uniform_(self.lora_As, a=math.sqrt(5))
        nn.init.zeros_(self.lora_Bs)

    def forward(self, hidden_states):
        intermediate = self.original_layer.wi(hidden_states)
        lora_output = hidden_states @ self.lora_As.T @ self.lora_Bs.T
        intermediate += lora_output
        intermediate = self.original_layer.act(intermediate)

        return self.original_layer.wo(intermediate)

class Router(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=1, beta=0.5):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        self.expert = nn.Parameter(torch.randn(num_experts, input_dim))
        self.top_k = top_k
        self.beta = beta
        

    def forward(self, hidden_states):
        # Gating score
        logits = self.gate(hidden_states)
        gate_score = F.softmax(logits, dim=-1)
        
        # Orthogonal score
        x_norm = F.normalize(hidden_states, dim=-1)
        e_norm = F.normalize(self.expert, dim=-1)
        cosine_sim = torch.matmul(x_norm, e_norm.T).abs()
        ortho_score = 1 - cosine_sim

        # Combined score
        score_mix = self.beta * gate_score + (1 - self.beta) * ortho_score
        
        return score_mix
    
    # 還有蟲
    def task_weight(self, dataset):
        self.eval()
        scores = []
        with torch.no_grad():

            for x in dataset:
                input_ids = x['input_ids'].unsqueeze(0).to(self.expert.device)  # (1, seq_len)
                attention_mask = x['attention_mask'].unsqueeze(0).to(self.expert.device)

                # 使用模型編碼器產生 hidden states
                encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = encoder_outputs.last_hidden_state  # (1, seq_len, hidden_dim)

                # 使用 [CLS]-like 向量（第一個 token）
                x_vec = hidden_states[:, 0, :]  # (1, hidden_dim)

                # 傳入 Router 計算 gating + orthogonal score
                score = self.forward(x_vec)  # (1, num_experts)
                scores.append(score.squeeze(0))  # (num_experts,)
            # for x in dataset: 
            #     x = x.unsqueeze(0).unsqueeze(0)
            #     S = self.forward(x)       
            #     scores.append(S.squeeze(0).squeeze(0)) 
        scores = torch.stack(scores)
        w_t = scores.mean(dim=0)

        # 選擇 top-k
        if self.top_k == 1:
            top_k_experts = torch.argmax(w_t, dim=-1)
        else:
            top_k_values, top_k_indices = torch.topk(w_t, k=self.top_k, dim=-1)
            top_k_experts = (top_k_indices, top_k_values)

        return top_k_experts
    
class MoEBlock(nn.Module):
    def __init__(self, original_layer, num_experts=4, rank=4, top_k=2):
        super().__init__()
        self.router = Router(original_layer.wi.weight.size(1), num_experts, top_k=top_k)
        self.experts = nn.ModuleList([LoRALayer(original_layer, rank=rank) for _ in range(num_experts)])
        self.selection_counts = torch.zeros(num_experts, dtype=torch.long, device=original_layer.wi.weight.device)
        self.task_top_k_experts = None
        self.last_scores = None
    
    def set_task_top_k(self, top_k_experts):
        self.task_top_k_experts = top_k_experts

    def forward(self, hidden_states):
        if self.task_top_k_experts is None:
            raise RuntimeError("[錯誤] 尚未指定 MoEBlock 的 top-k 專家。")

        top_k_experts = self.task_top_k_experts
        outputs = torch.zeros_like(hidden_states)
        used_experts = set()
        
        # if self.task_top_k_experts is not None:
        #     top_k_experts = self.task_top_k_experts
        #     scores = None
        # else:
        #     raise RuntimeError("嘗試使用非 Router.task_weight(dataset) 計算任務的 top-k 專家")
        # outputs = torch.zeros_like(hidden_states)

        for i, expert in enumerate(self.experts):
            if isinstance(top_k_experts, tuple):  # top-k 模式
                top_k_indices, top_k_values = top_k_experts  # (batch_size, seq_len, top_k)

                # 如果這個 expert 不在 top-k 中，就跳過
                if (top_k_indices == i).sum() == 0:
                    continue 

                used_experts.add(i)

                # 讓 `mask` 形狀符合 `expert_output`
                mask = (top_k_indices == i).to(hidden_states.device).float()  # (batch_size, seq_len, top_k)
                
                # **修正點：將 `top_k` 維度壓縮**
                mask = mask.sum(dim=-1, keepdim=True)  # -> (batch_size, seq_len, 1)

                # 確保 `mask` 形狀與 `hidden_states` 匹配
                mask = mask.expand_as(hidden_states)  # -> (batch_size, seq_len, hidden_dim)

                # 讓 `top_k_values` 維度與 `hidden_states` 相匹配
                top_k_values = top_k_values.sum(dim=-1, keepdim=True)  # -> (batch_size, seq_len, 1)
                mask = mask * top_k_values.expand_as(hidden_states)  # (batch_size, seq_len, hidden_dim)

            else:  # 單一選擇模式

                if (top_k_experts == i).sum() == 0:
                    continue

                used_experts.add(i)

                mask = (top_k_experts == i).to(hidden_states.device).float()
                mask = mask.unsqueeze(-1).expand_as(hidden_states)

            expert_output = expert(hidden_states)
            outputs += mask * expert_output

            # ✅ 修正 `selection_counts`
            self.selection_counts[i] += int(mask.sum().item())  # 避免 long() 轉換錯誤
        
        if len(used_experts) == 0:
            print("[警告] 沒有 expert 被選中參與參數更新。")
        else:
            print(f"[MoEBlock] 本輪參與參數更新的 experts: {sorted(used_experts)}")

        return outputs