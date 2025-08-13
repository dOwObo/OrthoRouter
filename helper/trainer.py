# helper/trainer.py
import torch
from torch.optim import AdamW
from transformers.optimization import Adafactor
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler
from tqdm import tqdm
from helper.utils import evaluate, visualize_expert_selection
# from helper.orth_router import ortho_regularization
from model.layers import MoEBlock, Router, LoRALayer
import logging
import matplotlib.pyplot as plt
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_balance_loss(scores):
    expert_mean_usage = scores.mean(dim=0)
    num_experts = scores.size(-1)
    balance_target = 1.0 / num_experts
    return ((expert_mean_usage - balance_target)**2).mean()

class Trainer:

    def __init__(
        self, 
        model, 
        train_dataloader, 
        eval_dataloader=None, 
        tokenizer=None, 
        labels_list=None, 
        device=None
    ):
        """
        初始化 Trainer。
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.labels_list = labels_list
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)

        # 初始化列表以存儲損失和準確率
        self.train_losses = []
        self.val_accuracies = []

    def train(
        self, 
        num_epochs=3, 
        learning_rate=1e-4, 
        output_dir="./", 
        accumulation_steps=1,
        max_grad_norm=1.0,
        lambda_orth=0.5,   # <-- 正交懲罰係數
        lambda_l2=0,     # <-- L2正則化係數
        beta=0.1
    ):
        """
        執行訓練過程，包含混合精度與梯度累積，並做梯度裁剪防止 NaN。
        """
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            scale_parameter=False, 
            relative_step=False, 
            lr=learning_rate
        )
        num_training_steps = len(self.train_dataloader) * num_epochs
        scheduler = get_scheduler(
            "linear", 
            optimizer=optimizer, 
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )
        
        scaler = GradScaler()
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                try:
                    outputs = self.model(
                        input_ids=batch["input_ids"], 
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    main_loss = outputs.loss  # 原始 loss

                    # 計算 LoRA 的正交懲罰
                    orth_l2_loss = sum(
                        module.compute_orth_loss(lambda_orth) for module in self.model.modules() if isinstance(module, LoRALayer)
                    )

                    # 計算 MoE 專家均衡損失
                    lb_loss = sum(
                        load_balance_loss(module.last_scores) for module in self.model.modules() if isinstance(module, MoEBlock)
                    )

                    # 總損失
                    loss = main_loss
                    # print("main_loss",main_loss)
                    # print("orth_l2_loss",orth_l2_loss)
                    # 考慮梯度累積
                    loss = loss / accumulation_steps
                    scaler.scale(loss).backward()

                    # 梯度累積處理
                    if (step + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()

                except Exception as e:
                    print(f"[ERROR] Step {step} encountered an error: {e}")
                    print(f"Input IDs: {batch['input_ids']}")
                    continue

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dataloader)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

            # 驗證
            if self.eval_dataloader is not None:
                accuracy = self.validate()
                self.val_accuracies.append(accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.save_model(output_dir, f"best_model_epoch_{epoch + 1}.bin")
            else:
                print("No eval_dataloader provided, skipping validation.")

        selection_counts = self.collect_selection_counts()
        print("selection_counts", selection_counts)
        visualize_expert_selection(selection_counts, output_dir=output_dir)
        print(f"Best Accuracy Achieved: {best_accuracy:.4f}")
        self.plot_metrics(output_dir)
        
    def add_new_task(self, new_dataloader):
        """
        當新任務來時，為 LoRA 和 MoE 參數動態增加新參數
        """
        self.logger.info("Adding new task: Expanding LoRA and MoE parameters...")

        # 為所有 LoRALayer 增加新參數
        for module in self.model.modules():
            if isinstance(module, LoRALayer):
                module.add_new_task()

        # 為所有 MoEBlock 的專家層也新增 LoRA
        for module in self.model.modules():
            if isinstance(module, MoEBlock):
                module.add_new_task()

        # 更新訓練數據
        self.train_dataloader = new_dataloader
        self.logger.info("New task added successfully.")

    def plot_metrics(self, output_dir):
        """
        繪製訓練損失和驗證準確率的變化圖。
        """
        epochs = range(1, len(self.train_losses) + 1)

        # 創建一個圖形
        plt.figure(figsize=(12, 5))
        train_losses = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss 
                for loss in self.train_losses]

        # 繪製訓練損失
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)

        # 繪製驗證準確率
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy over Epochs')
        plt.xticks(epochs)  # 設置 x 軸刻度為 epoch 數
        plt.legend()
        plt.grid(True)

        # 調整佈局
        plt.tight_layout()

        # 保存圖形
        plot_path = os.path.join(output_dir, 'training_metrics.png')
        plt.savefig(plot_path)
        plt.close()

        self.logger.info(f"Training metrics plot saved to {plot_path}")
        
    def validate(self):
        """
        在驗證集上評估模型 (evaluate)。
        """
        print("Evaluating...")
        
        if self.eval_dataloader is None:
            print("No eval_dataloader provided.")
            return 0.0

        accuracy = evaluate(self.model, self.eval_dataloader, self.tokenizer, self.labels_list)
        return accuracy

    def save_model(self, output_dir, model_name):
        """
        保存模型。
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


    def collect_selection_counts(self):
        selection_counts = []
        for name, module in self.model.named_modules():
            if isinstance(module, MoEBlock):
                counts = module.selection_counts.cpu().numpy().tolist()
                selection_counts.append(counts)
                module.selection_counts.zero_()  # 重置計數
        return selection_counts