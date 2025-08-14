# main.py
# nohup python main.py --seed 438 --save_dir ./saved_models> 0510.log 2>&1 &
import os
import shutil
import argparse
import logging
import json
import csv

import torch
from transformers import set_seed

from model.custom_t5 import CustomT5Model
from dataset.data_processor import DataProcessor
from helper.utils import collate_fn
from helper.trainer import Trainer
from model.layers import LoRALayer, MoEBlock, Router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_custom_model(load_directory, device):
    """
    自訂加載流程：加載 state_dict 和自定義配置，並初始化模型。
    """
    custom_model = CustomT5Model.load_pretrained(load_directory, device=device)
    logger.info(f"Model loaded from {load_directory}")
    return custom_model

def save_custom_model(custom_model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # 保存 state_dict
    state_dict_path = os.path.join(output_dir, "model_state_dict.pt")
    torch.save(custom_model.model.state_dict(), state_dict_path)
    logger.info(f"Model state_dict saved to {state_dict_path}")
    # 保存自定義 config
    config = {"num_experts": custom_model.num_experts, "expert_rank": custom_model.expert_rank}
    with open(os.path.join(output_dir, "custom_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    logger.info(f"Custom config saved to {output_dir}/custom_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task Continual Learning with T5-MoE & LoRA Expansion")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Base directory to save models')
    # 允許 shell 傳入 order 與自定義 datasets（可選）
    parser.add_argument('--datasets', type=str, default=None, help='Comma-separated dataset list to override the default order')
    args = parser.parse_args()

    # 設定隨機種子
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 任務與資料集配置
    datasets = ["dbpedia","amazon","yahoo","agnews"]
    # if args.datasets:
    # datasets = ["dbpedia","amazon","yahoo","agnews"]
    dataset_task_map = {"dbpedia": "TC", "amazon": "SC", "yahoo": "TC", "agnews": "TC"}
    base_data_dir = "./CL_Benchmark"
    base_model_path = "./initial_model/t5-large"
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    # 用於累積測試集
    test_data_files = []
    test_labels_files = []

    custom_model = None
    trainer = None

    for idx, dataset in enumerate(datasets):
        task = dataset_task_map[dataset]
        logger.info(f"=== Processing task {idx+1}/{len(datasets)}: {dataset} ({task}) ===")

        # 準備資料路徑
        data_file   = os.path.join(base_data_dir, task, dataset, 'train.json')
        labels_file = os.path.join(base_data_dir, task, dataset, 'labels.json')
        eval_file   = os.path.join(base_data_dir, task, dataset, 'dev.json')
        eval_labels = labels_file  # 相同 labels.json
        test_file   = os.path.join(base_data_dir, task, dataset, 'test.json')
        test_label  = labels_file

        # 累積測試集列表
        test_data_files.append(test_file)
        test_labels_files.append(test_label)

        # 建立 DataLoader
        train_proc = DataProcessor(data_file, labels_file, base_model_path, max_input_length=256, max_label_length=50)
        train_ds   = train_proc.get_dataset()
        train_dl   = train_proc.get_dataloader(train_ds, batch_size=8, collate_fn=collate_fn)
        #DEL 創建訓練集的 100 筆子集
        train_subset_dataloader = train_proc.get_subset_dataloader(
            train_ds,
            batch_size=8,
            collate_fn=collate_fn,
            subset_size=100,
            shuffle=True
        )

        eval_proc = DataProcessor(eval_file, eval_labels, base_model_path, max_input_length=256, max_label_length=50)
        eval_ds   = eval_proc.get_dataset()
        eval_dl   = eval_proc.get_dataloader(eval_ds, batch_size=8, collate_fn=collate_fn)
        #DEL 創建驗證集的 100 筆子集
        eval_subset_dataloader = eval_proc.get_subset_dataloader(
            eval_ds,
            batch_size=8,
            collate_fn=collate_fn,
            subset_size=100,
            shuffle=True
        )

        # 獲取任務ID
        task_id = train_proc.get_task_from_dataset(train_proc.dataset_name)
        task_id_mapping = {"SC": 0, "TC": 1, "NLI": 2, "QQP": 3, "WiC": 4, "MultiRC": 5, "COPA": 6, "BoolQA": 7}
        task_id = task_id_mapping.get(task_id, 0)
        
        print(f"[Main] 當前任務: {train_proc.task}, 任務ID: {task_id}")

        # 載入或初始化模型
        if idx == 0:
            logger.info("Initializing new model and trainer...")
            
            # 初始化 CustomT5Model
            custom_model = CustomT5Model(base_model_path, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), num_experts=4, expert_rank=8)
            model = custom_model.model

            # 計算 top-k 專家並設定給所有 MoEBlock
            hidden_dim = custom_model.model.config.d_model
            
            # 初始化 Router 並為新任務初始化專家
            router = Router(hidden_dim, num_experts=4, top_k=2)
            # 確保 Router 在正確的裝置上
            router = router.to(device)

            # 凍結非-LoRA/MoE 參數
            model.config.use_cache = False
            logger.info("凍結除了 LoRA/MoE 以外的參數...")
            for p in model.parameters(): p.requires_grad = False
            
            for name, module in model.named_modules():
                if isinstance(module, LoRALayer):
                    module.lora_As.requires_grad = True
                    module.lora_Bs.requires_grad = True
                elif isinstance(module, MoEBlock):
                    for expert in module.experts:
                        for param in expert.parameters():
                            param.requires_grad = True  # MoE 專家層可訓練
            logger.info("Parameter freezing completed.")

            trainer = Trainer(
                model=model, 
                train_dataloader=train_dl, 
                eval_dataloader=eval_dl,
                # train_dataloader=train_subset_dataloader,
                # eval_dataloader=eval_subset_dataloader,
                tokenizer=train_proc.tokenizer, 
                labels_list=train_proc.labels_list,
                device=device,
                task_id=task_id  # 傳入任務ID
            )
        else:
            logger.info("Adding new task (dynamic LoRA & MoE expansion)...")
            
            # 只更新 trainer 的 dataloader
            trainer.train_dataloader = train_dl
            trainer.eval_dataloader = eval_dl
            
        # 為新任務初始化專家向量
        router.initialize_expert_for_task(task_id, train_ds, custom_model.model)
        
        # 使用改進的 task_weight 方法計算 top-k 專家
        topk_experts = router.task_weight(
            dataset=train_ds, 
            encoder_model=custom_model.model, 
            task_id=task_id, 
            strategy='confident'  # 使用置信度篩選策略
        )
        
        # 設定所有 MoEBlock 的 top-k 專家和任務ID
        for name, module in model.named_modules():
            if isinstance(module, MoEBlock):
                module.initialize_task_experts(task_id, train_ds, custom_model.model)
                module.set_task_top_k(topk_experts, task_id)
        
        logger.info(f"✅ 已為任務 {task_id} 設定所有 MoEBlock 的 top-k 專家。")

        # 列印每層 LoRA 子空間數量
        # for layer_idx, module in enumerate(model.modules()):
        #     if isinstance(module, LoRALayer):
        #         print(f"[Layer {layer_idx}] total LoRA tasks: {len(module.lora_As)}")

        # 開始訓練
        output_subdir = os.path.join(save_dir, dataset)
        os.makedirs(output_subdir, exist_ok=True)
        trainer.train(num_epochs=3, learning_rate=5e-4, output_dir=output_subdir, accumulation_steps=64)

        # 保存模型與自定義配置
        save_custom_model(custom_model, output_subdir)

        # 評估所有已累積測試集
        logger.info("Evaluating on all seen test sets...")
        # 針對本 dataset 的測試結果 CSV（按 seed 匯總）
        test_csv_path = os.path.join(output_subdir, f'test_results_seed_{args.seed}.csv')
        if not os.path.exists(test_csv_path):
            with open(test_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['seed', 'tested_dataset', 'accuracy'])
        for td, tl in zip(test_data_files, test_labels_files):
            logger.info(f"Testing {td}...")
            test_proc = DataProcessor(td, tl, base_model_path, max_input_length=256, max_label_length=50)
            test_ds   = test_proc.get_dataset()
            test_dl   = test_proc.get_dataloader(test_ds, batch_size=8, collate_fn=collate_fn)
            # DEL 創建測試集的 100 筆子集
            test_subset_dataloader = test_proc.get_subset_dataloader(
                test_ds,
                batch_size=8,
                collate_fn=collate_fn,
                subset_size=100,
                shuffle=True
            )
            trainer.eval_dataloader = test_dl
            # trainer.eval_dataloader = test_subset_dataloader
            acc = trainer.validate()
            logger.info(f"Test Accuracy on {td}: {acc:.4f}")
            # 寫入測試結果
            with open(test_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([args.seed, os.path.basename(td), float(acc)])

    logger.info("All tasks completed.")
