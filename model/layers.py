import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank: int = 4):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank

        # 固定 LoRA 參數，不進行動態擴展
        self.lora_As = nn.Parameter(torch.zeros((rank, original_layer.wi.weight.size(1))))
        self.lora_Bs = nn.Parameter(torch.zeros((original_layer.wi.weight.size(0), rank)))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_As, a=math.sqrt(5))
        nn.init.zeros_(self.lora_Bs)
    
    def add_new_task(self):
        # 固定模式下不做任何事情
        pass

    def forward(self, hidden_states):
        intermediate = self.original_layer.wi(hidden_states)
        lora_output = hidden_states @ self.lora_As.T @ self.lora_Bs.T
        intermediate += lora_output
        intermediate = self.original_layer.act(intermediate)

        return self.original_layer.wo(intermediate)

    def compute_orth_loss(self, lambda_orth=0.5):
        """
        計算 LoRA 參數的正交懲罰損失
        用於確保 LoRA 矩陣的正交性，避免過擬合
        """
        # 計算 A 矩陣的正交懲罰
        A_orth_loss = torch.norm(torch.mm(self.lora_As, self.lora_As.T) - torch.eye(self.lora_As.size(0), device=self.lora_As.device))
        
        # 計算 B 矩陣的正交懲罰
        B_orth_loss = torch.norm(torch.mm(self.lora_Bs.T, self.lora_Bs) - torch.eye(self.lora_Bs.size(1), device=self.lora_Bs.device))
        
        # 總正交損失
        total_orth_loss = lambda_orth * (A_orth_loss + B_orth_loss)
        
        return total_orth_loss

class Router(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=1, beta=0.5):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        self.expert = nn.Parameter(torch.randn(num_experts, input_dim))
        self.top_k = top_k  # 支持 top-k
        self.beta = beta

        # 使用普通的字典來管理任務嵌入，並手動註冊為 Parameter
        self.task_embeddings = {}
        self.initialized_experts = set()  # 已初始化的專家集合

    def initialize_expert_for_task(self, task_id, dataset, encoder_model):
        """
        為新任務初始化專家向量，使用任務數據的語義中心
        """
        if task_id in self.initialized_experts:
            print(f"[Router] 任務 {task_id} 的專家已初始化，跳過")
            return
            
        print(f"[Router] 為任務 {task_id} 初始化專家向量...")
        
        # 計算任務數據的語義中心
        task_centroid = self.get_task_centroid(dataset, encoder_model)
        
        # 找到未初始化的專家位置
        available_expert_idx = None
        for i in range(self.expert.size(0)):
            if i not in self.initialized_experts:
                available_expert_idx = i
                break
        
        if available_expert_idx is not None:
            # 用任務中心初始化專家向量
            with torch.no_grad():
                self.expert[available_expert_idx] = task_centroid.detach().clone()
            self.initialized_experts.add(available_expert_idx)
            
            # 創建任務embedding - 使用正確的方式註冊 Parameter
            task_id_str = str(task_id)
            if task_id_str not in self.task_embeddings:
                # 創建新的 Parameter 並手動註冊
                task_embedding = nn.Parameter(task_centroid.detach().clone())
                self.task_embeddings[task_id_str] = task_embedding
                # 手動註冊為模組的參數
                self.register_parameter(f'task_embedding_{task_id_str}', task_embedding)
                
            print(f"[Router] 專家 {available_expert_idx} 已用任務 {task_id} 的中心向量初始化")
        else:
            print(f"[Router] 警告：沒有可用的專家位置為任務 {task_id} 初始化")

    def get_task_centroid(self, dataset, encoder_model):
        """
        計算任務數據的語義中心向量
        使用簡化的方法避免 MoEBlock 未初始化問題
        """
        # 確保 encoder_model 在正確的裝置上
        device = self.expert.device
        encoder_model = encoder_model.to(device)
        encoder_model.eval()
        
        centroids = []
        
        with torch.no_grad():
            for x in dataset:
                # 直接在正確裝置上創建 tensor
                input_ids = torch.tensor(x['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
                attention_mask = torch.tensor(x['attention_mask'], dtype=torch.long, device=device).unsqueeze(0)
                
                # 方法1：只使用 embedding 層（最簡單，避免 MoE 問題）
                # 這是一個簡化的語義表示，雖然不如完整 encoder 豐富，但足夠用於初始化
                hidden_states = encoder_model.encoder.embed_tokens(input_ids)
                
                # 使用平均池化而不是 [CLS]
                # 因為 T5 沒有 [CLS] token，我們使用所有 token 的平均
                mask = attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
                masked_hidden = hidden_states * mask
                avg_hidden = masked_hidden.sum(dim=1) / mask.sum(dim=1)  # (1, hidden_dim)
                
                centroids.append(avg_hidden.squeeze(0))
        
        # 計算平均中心向量
        task_centroid = torch.stack(centroids).mean(dim=0)
        print(f"[Router] 計算完成任務語義中心，使用 {len(centroids)} 個樣本")
        return task_centroid

    def forward(self, hidden_states, task_id=None):
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

    def task_weight(self, dataset, encoder_model=None, task_id=None, strategy='mean'):
        """
        改進的任務權重計算，支援多種聚合策略
        使用簡化的方法避免 MoEBlock 未初始化問題
        """
        # 確保 encoder_model 在正確的裝置上
        device = self.expert.device
        encoder_model = encoder_model.to(device)
        
        self.eval()
        scores = []
        confidences = []
        
        with torch.no_grad():
            for x in dataset:
                # 直接在正確裝置上創建 tensor
                input_ids = torch.tensor(x['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
                attention_mask = torch.tensor(x['attention_mask'], dtype=torch.long, device=device).unsqueeze(0)

                # 使用簡化的方法：只使用 embedding 層
                hidden_states = encoder_model.encoder.embed_tokens(input_ids)
                
                # 使用平均池化
                mask = attention_mask.unsqueeze(-1).float()
                masked_hidden = hidden_states * mask
                x_vec = masked_hidden.sum(dim=1) / mask.sum(dim=1)  # (1, hidden_dim)

                # 傳入 Router 計算 gating + orthogonal score
                score = self.forward(x_vec, task_id)
                
                # 計算置信度（基於分數的熵）
                entropy = -torch.sum(score * torch.log(score + 1e-8), dim=-1)
                confidence = 1.0 - entropy / torch.log(torch.tensor(score.size(-1), dtype=torch.float, device=device))
                
                scores.append(score.squeeze(0))
                confidences.append(confidence.squeeze(0))

        scores = torch.stack(scores)  # (num_samples, num_experts)
        confidences = torch.stack(confidences)  # (num_samples,)
        
        # 根據策略聚合分數
        if strategy == 'mean':
            w_t = scores.mean(dim=0)
        elif strategy == 'confident':
            # 只使用高置信度的樣本
            confident_mask = confidences > 0.7
            if confident_mask.sum() > 0:
                w_t = scores[confident_mask].mean(dim=0)
            else:
                w_t = scores.mean(dim=0)
        elif strategy == 'top_mean':
            # 使用前50%高分數樣本的平均
            num_top = max(1, scores.size(0) // 2)
            top_scores, _ = torch.topk(scores, k=num_top, dim=0)
            w_t = top_scores.mean(dim=0)
        else:
            w_t = scores.mean(dim=0)

        # 選擇 top-k
        if self.top_k == 1:
            top_k_experts = torch.argmax(w_t, dim=-1)
        else:
            top_k_values, top_k_indices = torch.topk(w_t, k=self.top_k, dim=-1)
            top_k_experts = (top_k_indices, top_k_values)

        # 調試信息 - 只在第一次調用時輸出
        if not hasattr(self, '_debug_printed'):
            print(f"[Router] task_weight 返回的 top_k_experts 形狀:")
            if isinstance(top_k_experts, tuple):
                print(f"  top_k_indices: {top_k_experts[0].shape}")
                print(f"  top_k_values: {top_k_experts[1].shape}")
            else:
                print(f"  top_k_experts: {top_k_experts.shape}")
            print(f"[Router] 選擇的專家: {top_k_experts}")
            self._debug_printed = True

        return top_k_experts
    
class MoEBlock(nn.Module):
    def __init__(self, original_layer, num_experts=4, rank=4, top_k=2):
        super().__init__()
        self.router = Router(original_layer.wi.weight.size(1), num_experts, top_k=top_k)
        self.experts = nn.ModuleList([LoRALayer(original_layer, rank=rank) for _ in range(num_experts)])
        self.selection_counts = torch.zeros(num_experts, dtype=torch.long, device=original_layer.wi.weight.device)
        self.last_scores = None
        self.task_top_k_experts = None
        self.current_task_id = None

    def initialize_task_experts(self, task_id, dataset, encoder_model):
        """
        為新任務初始化專家
        """
        self.router.initialize_expert_for_task(task_id, dataset, encoder_model)
        self.current_task_id = task_id
    
    def set_task_top_k(self, top_k_experts, task_id=None):
        self.task_top_k_experts = top_k_experts
        if task_id is not None:
            self.current_task_id = task_id

    def forward(self, hidden_states):
        if self.task_top_k_experts is None:
            raise RuntimeError("[錯誤] 尚未指定 MoEBlock 的 top-k 專家。")
        
        top_k_experts = self.task_top_k_experts
        outputs = torch.zeros_like(hidden_states)
        used_experts = set()

        # 獲取 batch 和 seq 的維度
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 計算路由分數（用於 load balance loss）
        # 使用平均池化計算每個樣本的路由分數
        avg_hidden = hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
        router_scores = self.router.forward(avg_hidden)  # (batch_size, num_experts)
        self.last_scores = router_scores  # 存儲用於 load balance loss

        for i, expert in enumerate(self.experts):
            if isinstance(top_k_experts, tuple):  # top-k 模式
                top_k_indices, top_k_values = top_k_experts  # (batch_size, seq_len, top_k)

                # 檢查這個 expert 是否在 top-k 中
                # top_k_indices 形狀: (batch_size, seq_len, top_k)
                expert_selected = (top_k_indices == i).any(dim=-1)  # (batch_size, seq_len)
                
                if not expert_selected.any():
                    continue 

                used_experts.add(i)

                # 創建正確的 mask
                # expert_selected: (batch_size, seq_len)
                mask = expert_selected.to(hidden_states.device).float()  # (batch_size, seq_len)
                mask = mask.unsqueeze(-1).expand_as(hidden_states)  # (batch_size, seq_len, hidden_dim)

                # 計算權重：如果這個 expert 被選中，使用對應的權重
                expert_weights = (top_k_indices == i).float() * top_k_values  # (batch_size, seq_len, top_k)
                expert_weight = expert_weights.sum(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
                expert_weight = expert_weight.expand_as(hidden_states)  # (batch_size, seq_len, hidden_dim)
                
                # 應用權重
                mask = mask * expert_weight

            else:  # 單一選擇模式
                expert_selected = (top_k_experts == i)
                
                if not expert_selected.any():
                    continue

                used_experts.add(i)

                mask = expert_selected.to(hidden_states.device).float()
                mask = mask.unsqueeze(-1).expand_as(hidden_states)

            expert_output = expert(hidden_states)
            outputs += mask * expert_output

            # ✅ 修正 `selection_counts`
            self.selection_counts[i] += int(mask.sum().item())  # 避免 long() 轉換錯誤

        # 移除高頻日誌輸出，只在沒有專家被選中時警告
        if len(used_experts) == 0:
            print("[警告] 沒有 expert 被選中參與參數更新。")

        return outputs