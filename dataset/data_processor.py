#  dataset/data_processor.py
import os
import json
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 之後刪除，測試用
from torch.utils.data import DataLoader, Subset
import random


class DataProcessor:
    def __init__(self, 
                 data_file: str, 
                 labels_file: str, 
                 peft_model_path: str, 
                 max_input_length: int = 512, 
                 max_label_length: int = 32,
                 config_dir: str = "configs"):
        """
        初始化資料處理流程。
        """
        self.data_file = data_file
        self.labels_file = labels_file
        self.peft_model_path = peft_model_path
        self.max_input_length = max_input_length
        self.max_label_length = max_label_length
        self.config_dir = config_dir

        # 讀取該資料集的提示的文件
        self.task_config = self.load_json(os.path.join(self.config_dir, "task.json"))
        self.instruction_config = self.load_json(os.path.join(self.config_dir, "instruction_config.json"))

        # 讀取資料
        self.data_df = pd.read_json(self.data_file)

        # 讀取標籤並建立映射
        with open(self.labels_file, "r", encoding="utf-8") as f:
            self.labels_list = json.load(f)
        # 將標籤轉換為 Snake_Case
        self.labels_list = [label.replace(" ", "_") for label in self.labels_list]
        self.label_to_id = {label: idx for idx, label in enumerate(self.labels_list)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.labels_list)}

        # 取得 `dataset/` 所在的根目錄（`OrthMoE/`）
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 設定 `initial_model/t5-large/` 的完整路徑
        base_model_path = os.path.join(ROOT_DIR, "initial_model/t5-large")

        # **確保 `config.json` 存在，否則改用 `initial_model/t5-large/`**
        if os.path.exists(os.path.join(self.peft_model_path, "config.json")):
            config_path = self.peft_model_path
        else:
            logger.warning(f"⚠️ 警告: `{self.peft_model_path}` 缺少 `config.json`，改用 `{base_model_path}`")
            config_path = base_model_path


        logger.info(f"🔄 使用 tokenizer 設定檔: {config_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config_path)
        self.tokenizer.padding_side = "right"

        # 確定數據集名稱和任務
        self.dataset_name = self.get_dataset_name(self.data_file)
        self.task = self.get_task_from_dataset(self.dataset_name)
        self.instruction = self.get_instruction(self.task)
        logger.info(f"Dataset: {self.dataset_name}, Task: {self.task}")

    def load_json(self, filepath):
        """
        加載 JSON 文件。
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"解析 JSON 文件失敗: {filepath}, 錯誤: {e}")
            raise

    def load_data(self, filepath):
        """
        加載數據文件。
        """
        try:
            data_df = pd.read_json(filepath)
            return data_df
        except ValueError as e:
            logger.error(f"讀取數據文件時出錯: {filepath}, 錯誤: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"數據文件不存在: {filepath}")
            raise

    def load_labels(self, filepath):
        """
        加載標籤文件並轉換為 Snake_Case。
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                labels = json.load(f)
            # 將標籤轉換為 Snake_Case
            labels = [label.replace(" ", "_") for label in labels]
            if len(labels) != len(set(labels)):
                logger.error("轉換後的標籤存在重複，請檢查標籤文件。")
                raise ValueError("轉換後的標籤存在重複，請檢查標籤文件。")
            return labels
        except FileNotFoundError:
            logger.error(f"標籤文件不存在: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"解析標籤 JSON 文件失敗: {filepath}, 錯誤: {e}")
            raise

    def get_dataset_name(self, data_file):
        """
        從 data_file 路徑中提取數據集名稱。
        假設路徑格式包含 dataset name，例如 ./CL_Benchmark/TC/dbpedia/train.json
        """
        parts = data_file.split(os.sep)
        for part in parts:
            if part in self.get_all_datasets():
                return part
        logger.error(f"無法從數據文件路徑中提取數據集名稱: {data_file}")
        raise ValueError(f"無法從數據文件路徑中提取數據集名稱: {data_file}")

    def get_all_datasets(self):
        """
        從 task_config 中獲取所有數據集名稱。
        """
        datasets = []
        for task, dataset_list in self.task_config.items():
            for dataset in dataset_list:
                datasets.append(dataset["dataset name"])
        return datasets

    def get_task_from_dataset(self, dataset_name):
        """
        根據數據集名稱從 task_config 中獲取對應的任務。
        """
        for task, dataset_list in self.task_config.items():
            for dataset in dataset_list:
                if dataset["dataset name"] == dataset_name:
                    return task
        logger.error(f"未找到數據集名稱 '{dataset_name}' 對應的任務。")
        raise ValueError(f"未找到數據集名稱 '{dataset_name}' 對應的任務。")

    def get_instruction(self, task):
        """
        根據任務從 instruction_config 中獲取對應的指令。
        """
        if task not in self.instruction_config:
            logger.error(f"未找到任務 '{task}' 的指令。")
            raise ValueError(f"未找到任務 '{task}' 的指令。")
        instructions = self.instruction_config[task]
        if not instructions:
            logger.error(f"任務 '{task}' 的指令列表為空。")
            raise ValueError(f"任務 '{task}' 的指令列表為空。")
        # 假設每個任務只有一個指令，選擇第一個
        return instructions[0]["instruction"]

    def convert_label_to_id(self, example):
        """
        將文字標籤轉換為數字 ID。
        """
        if example["label"] in self.label_to_id:
            example["label_id"] = self.label_to_id[example["label"]]
        else:
            raise ValueError(f"未知標籤: {example['label']}，請檢查標籤文件是否包含此標籤。")
        return example

    def preprocess_data(self, example):
        """
        將資料轉換為提示式輸入。
        """
        try:
            options = ", ".join(self.labels_list)
            input_text = (
                f"Task:{self.task}\nDataset:{self.dataset_name}\n"
                f"{self.instruction}"
                f"Option: {options}\n"
                f"{example['sentence']}\nAnswer:"
            )
            # 把產生的 prompt 直接加進 example，方便後續 tokenize
            example["input_text"] = input_text
        except KeyError as e:
            logger.error(f"缺少必要的欄位: {e}")
            raise KeyError(f"缺少必要的欄位: {e}")
        return example

    def tokenize_data(self, examples):
        """
        Hugging Face Dataset 在 batched=True 時，examples 會是一批資料的 dict of list。
        我們要針對整批一起 tokenize，再拆分回去。
        """
        # 1. Tokenize input_text
        model_inputs = self.tokenizer(
            examples["input_text"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True
        )

        # 2. Tokenize labels (把 label_id 轉為真正的文字標籤後做 tokenize)
        label_texts = [self.labels_list[idx] for idx in examples["label_id"]]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                label_texts,
                max_length=self.max_label_length,
                padding="max_length",
                truncation=True
            )

        # 3. 建立 model_inputs["labels"]，並把 -100 mask 給 padding
        label_masks = labels["attention_mask"]
        label_ids = labels["input_ids"]
        # 把 padding token (= tokenizer.pad_token_id) 的位置換成 -100，避免影響 loss
        for i in range(len(label_ids)):
            for j in range(len(label_ids[i])):
                if label_masks[i][j] == 0:
                    label_ids[i][j] = -100

        # 加到 model_inputs
        model_inputs["labels"] = label_ids
        return model_inputs

    def get_dataset(self):
        """
        取得 Dataset（含前處理與標記化）。
        """
        # 檢查標籤有效性
        invalid_labels = [label for label in self.data_df["label"] if label not in self.label_to_id]
        if invalid_labels:
            # print(f"[WARNING] 資料包含無效標籤: {invalid_labels}")
            self.data_df = self.data_df[~self.data_df["label"].isin(invalid_labels)]

        # 檢查空內容
        invalid_sentences = self.data_df["sentence"].apply(lambda x: not x.strip())
        if invalid_sentences.any():
            print(f"[WARNING] 空句子被移除: {self.data_df[invalid_sentences].index.tolist()}")
            self.data_df = self.data_df[~invalid_sentences]

        # 轉換標籤為 ID
        self.data_df = self.data_df.apply(self.convert_label_to_id, axis=1)

        # 轉換成 Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(self.data_df)

        # 先進行 preprocess_data (單筆即可)
        hf_dataset = hf_dataset.map(self.preprocess_data)

        # 再進行 tokenize_data (建議 batched=True)
        hf_dataset = hf_dataset.map(
            self.tokenize_data,
            batched=True,
            remove_columns=["sentence", "label", "label_id", "input_text"]  
            # 避免把原本的 columns 帶進後面的 DataLoader
        )
        return hf_dataset

    def get_dataloader(self, dataset, batch_size, collate_fn):
        """
        根據 Dataset 建立 DataLoader。
        """
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn
        )
    # 之後刪除，測試用
    def get_subset_dataloader(self, dataset, batch_size, collate_fn, subset_size=500, shuffle=True):
        """
        從 dataset 中隨機抽取 subset_size 筆資料，並返回相應的 DataLoader。
        """
        dataset_size = len(dataset)
        if subset_size > dataset_size:
            raise ValueError(f"subset_size ({subset_size}) 大於資料集大小 ({dataset_size})")
        indices = list(range(subset_size))
        subset = Subset(dataset, indices)
        return self.get_dataloader(subset, batch_size, collate_fn)