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
        """當新任務來時，動態新增 LoRA 參數"""
        new_A = nn.Parameter(torch.zeros((self.rank, self.original_layer.wi.weight.size(1))))
        new_B = nn.Parameter(torch.zeros((self.original_layer.wi.weight.size(0), self.rank)))

        # 確保新參數與舊參數正交
        if len(self.lora_As) > 0:
            new_A.data = orthogonal_projection(self.lora_As[-1].detach(), new_A)

        nn.init.kaiming_uniform_(new_A, a=math.sqrt(5))
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

    # def forward(self, hidden_states):
    #     intermediate = self.original_layer.wi(hidden_states)

    #     # 遍歷所有 LoRA 參數（舊 + 新）
    #     for A, B in zip(self.lora_As, self.lora_Bs):
    #         lora_output = hidden_states @ A.T @ B.T
    #         intermediate += lora_output

    #     intermediate = self.original_layer.act(intermediate)
    #     return self.original_layer.wo(intermediate)

    def compute_orth_loss(self, lambda_orth: float) -> torch.Tensor:
        """計算 LoRA 參數的正交懲罰，防止新參數干擾舊參數"""
        loss = 0
        for i in range(len(self.lora_As) - 1):
            loss += torch.mm(self.lora_As[i], self.lora_As[-1].T).pow(2).sum()
            loss += torch.mm(self.lora_Bs[i].T, self.lora_Bs[-1]).pow(2).sum()

            # loss += torch.abs(torch.mm(self.lora_As[i], self.lora_As[-1].T)).sum()
            # loss += torch.abs(torch.mm(self.lora_Bs[i].T, self.lora_Bs[-1])).sum()
        return lambda_orth * loss
    #    待修改torch.abs(torch.mm(self.lora_A, self.loranew_A.T)).sum()

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

# NOTE 依據正交性
# ------------------------------------------------
# 新策略：使用多組 LoRA 後的 Router
# ------------------------------------------------
# class Router(nn.Module):
#     """
#     修改後的 Router：
#     - 利用線性層計算 gate 得分。
#     - 另外計算專家知識向量與輸入之間的 cosine similarity，再轉換為正交性得分：ortho_score = 1 - |cosine_similarity|
#     - 融合兩者，進行專家選擇。
#     """
#     def __init__(self, input_dim, num_experts, experts=None):
#         super().__init__()
#         self.gate = nn.Linear(input_dim, num_experts)
#         self.experts = experts  # 用於計算正交性分數
#         self.alpha = nn.Parameter(torch.tensor(2.0))  # 正交性權重
#         self.beta  = nn.Parameter(torch.tensor(1.0))  # Gate 得分權重

#     # def compute_orthogonality_scores(self, hidden_states):
#     #     """
#     #     計算每個專家與輸入 hidden_states 的 cosine similarity，
#     #     並轉換為正交性得分: ortho_score = 1 - |cosine_similarity|
#     #     """
#     #     ortho_scores = []
#     #     for idx, expert in enumerate(self.experts):
#     #         # ★★★ 改用 lora_As[-1] 取得最後一組 LoRA 的 A
#     #         # 取 lora_As[-1] 的均值作為知識向量
#     #         knowledge_vector = expert.lora_As[-1].mean(dim=0)

#     #         # 正規化知識向量
#     #         knowledge_vector = knowledge_vector / (knowledge_vector.norm(p=2) + 1e-8)
#     #         # 正規化 hidden_states
#     #         hidden_norm = hidden_states / (hidden_states.norm(p=2, dim=-1, keepdim=True) + 1e-8)

#     #         # 計算 cosine similarity
#     #         raw_cosine = torch.sum(hidden_norm * knowledge_vector, dim=-1)
#     #         # 轉換為正交性得分：越接近正交 (raw_cosine 趨近 0)，得分越高
#     #         score = 1.0 - torch.abs(raw_cosine)
#     #         ortho_scores.append(score)

#     #     ortho_scores = torch.stack(ortho_scores, dim=-1)  # shape: [batch_size, num_experts]
#     #     return ortho_scores

#     def compute_orthogonality_scores(self, hidden_states):
#         """
#         計算每個專家與輸入 hidden_states 的 cosine similarity，
#         並轉換為正交性得分: ortho_score = 1 - |cosine_similarity|
#         這裡將所有過去任務的 LoRA 參數做平均，捕捉歷史累積知識。
#         """
#         ortho_scores = []
#         for idx, expert in enumerate(self.experts):
#             # 堆疊所有 lora_As，形狀為 (num_tasks, rank, hidden_dim)
#             stacked_As = torch.stack(list(expert.lora_As), dim=0)
#             # 先平均任務維度： shape 變成 (rank, hidden_dim)
#             mean_over_tasks = stacked_As.mean(dim=0)
#             # 再平均 rank 維度，得到 shape (hidden_dim,)
#             knowledge_vector = mean_over_tasks.mean(dim=0)
            
#             # 正規化知識向量
#             knowledge_vector = knowledge_vector / (knowledge_vector.norm(p=2) + 1e-8)
#             # 正規化 hidden_states
#             hidden_norm = hidden_states / (hidden_states.norm(p=2, dim=-1, keepdim=True) + 1e-8)
#             # 計算 cosine similarity
#             raw_cosine = torch.sum(hidden_norm * knowledge_vector, dim=-1)
#             # 轉換為正交性得分：越接近正交 (raw_cosine 趨近 0) 得分越高
#             score = 1.0 - torch.abs(raw_cosine)
#             ortho_scores.append(score)
            
#         ortho_scores = torch.stack(ortho_scores, dim=-1)  # shape: [batch_size, num_experts]
#         return ortho_scores



#     def forward(self, hidden_states):
#         # 計算 gate 得分 (線性層後 softmax)
#         gate_logits = self.gate(hidden_states)  # shape: [batch_size, num_experts]
#         gate_scores = torch.softmax(gate_logits, dim=-1)
        
#         if self.experts is not None:
#             ortho_scores = self.compute_orthogonality_scores(hidden_states)  # shape: [batch_size, num_experts]
#             combined_scores = self.beta * gate_scores + self.alpha * ortho_scores
#             combined_scores = torch.softmax(combined_scores, dim=-1)
#         else:
#             combined_scores = gate_scores

#         top_experts = torch.argmax(combined_scores, dim=-1)
#         return top_experts, combined_scores

    
class MoEBlock(nn.Module):
    def __init__(self, original_layer, num_experts=4, rank=4, top_k=2):
        super().__init__()
        self.router = Router(original_layer.wi.weight.size(1), num_experts, top_k=top_k)
        self.experts = nn.ModuleList([LoRALayer(original_layer, rank=rank) for _ in range(num_experts)])
        # 將 experts 傳遞給 Router
        # self.router = Router(
        #     input_dim=original_layer.wi.weight.size(1),
        #     num_experts=num_experts,
        #     experts=self.experts  # <--- 傳遞已建立的 experts
        # )
        self.selection_counts = torch.zeros(num_experts, dtype=torch.long, device=original_layer.wi.weight.device)
        self.last_scores = None
    def forward(self, hidden_states):
        top_k_experts, scores = self.router(hidden_states)
        self.last_scores = scores.detach()  # 避免梯度流入
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

