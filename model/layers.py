import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import math

def orthogonal_projection(A_old: torch.Tensor, A_new: torch.Tensor) -> torch.Tensor:
    """確保新 LoRA 參數與舊參數正交"""
    if A_old is None or A_old.numel() == 0:
        return A_new

    Q, _ = torch.linalg.qr(A_old.T)
    A_new_proj = A_new - (A_new @ Q) @ Q.T
    return A_new_proj

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank: int = 4):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank

        # LoRA 參數池，每個新任務來時會動態擴展
        self.lora_As = nn.ParameterList()
        self.lora_Bs = nn.ParameterList()

        # 初始化第一組 LoRA 參數
        self.add_new_task()

    def add_new_task(self):
        device = self.original_layer.wi.weight.device
        """當新任務來時，動態新增 LoRA 參數"""
        new_A = nn.Parameter(torch.zeros((self.rank, self.original_layer.wi.weight.size(1)), device=device))
        new_B = nn.Parameter(torch.zeros((self.original_layer.wi.weight.size(0), self.rank), device=device))

        nn.init.kaiming_uniform_(new_A, a=math.sqrt(5))
        # 確保新參數與舊參數正交
        if len(self.lora_As) > 0:
            A_old = self.lora_As[-1].detach().to(device)
            new_A.data = orthogonal_projection(A_old, new_A.data)
        nn.init.zeros_(new_B)

        self.lora_As.append(new_A)
        self.lora_Bs.append(new_B)
    def forward(self, hidden_states):
        intermediate = self.original_layer.wi(hidden_states)

        lora_total_norm = 0  # 紀錄 LoRA 的總影響力

        for i, (A, B) in enumerate(zip(self.lora_As, self.lora_Bs)):
            lora_output = hidden_states @ A.T @ B.T
            intermediate += lora_output

            # 確保 LoRA 參數有實際影響
            lora_norm = lora_output.norm().item()
            lora_total_norm += lora_norm

        intermediate = self.original_layer.act(intermediate)
        return self.original_layer.wo(intermediate)

    def compute_orth_loss(self, lambda_orth: float) -> torch.Tensor:
        """計算 LoRA 參數的正交懲罰，防止新參數干擾舊參數"""
        loss = 0
        for i in range(len(self.lora_As) - 1):
            loss += torch.mm(self.lora_As[i], self.lora_As[-1].T).pow(2).sum()
            loss += torch.mm(self.lora_Bs[i].T, self.lora_Bs[-1]).pow(2).sum()
        return lambda_orth * loss

class Router(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=1):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k  # 支持 top-k

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        scores = F.softmax(logits, dim=-1)

        if self.top_k == 1:
            top_k_experts = torch.argmax(scores, dim=-1)
        else:
            top_k_values, top_k_indices = torch.topk(scores, k=self.top_k, dim=-1)
            top_k_experts = (top_k_indices, top_k_values)

        return top_k_experts, scores


    
class MoEBlock(nn.Module):
    def __init__(self, original_layer, num_experts=4, rank=4, top_k=2):
        super().__init__()
        self.router = Router(original_layer.wi.weight.size(1), num_experts, top_k=top_k)
        self.experts = nn.ModuleList([LoRALayer(original_layer, rank=rank) for _ in range(num_experts)])
        self.selection_counts = torch.zeros(num_experts, dtype=torch.long, device=original_layer.wi.weight.device)
        self.last_scores = None
    def forward(self, hidden_states):
        top_k_experts, scores = self.router(hidden_states)
        self.last_scores = scores
        outputs = torch.zeros_like(hidden_states)

        for i, expert in enumerate(self.experts):
            if isinstance(top_k_experts, tuple):  # top-k 模式
                top_k_indices, top_k_values = top_k_experts  # (batch_size, seq_len, top_k)

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
                mask = (top_k_experts == i).to(hidden_states.device).float()
                mask = mask.unsqueeze(-1).expand_as(hidden_states)

            expert_output = expert(hidden_states)
            outputs += mask * expert_output

            # ✅ 修正 `selection_counts`
            self.selection_counts[i] += int(mask.sum().item())  # 避免 long() 轉換錯誤

        return outputs


    def add_new_task(self):
        """當新任務來時，所有專家的 LoRA 都會新增一組參數"""
        for expert in self.experts:
            expert.add_new_task()

