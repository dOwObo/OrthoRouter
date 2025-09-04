## NLP Task2Vec: 任務嵌入與相似度分析

**NLP Task2Vec** 是一個實作 **Task Embedding** 概念的專案，它能將不同的 NLP 分類任務轉換成一個高維向量（即**任務向量**），並藉由這些向量計算任務之間的相似度。這個專案的核心在於利用 **Fisher 資訊矩陣 (FIM)**，來捕捉每個任務在預訓練模型（如 RoBERTa）上訓練一個輕量級分類頭時的梯度資訊。

---

## 核心概念

* **任務嵌入（Task Embedding）**：將一個 NLP 任務（例如：情感分析、主題分類）表示為一個向量。向量距離越近，代表任務越相似。
* **Fisher 資訊矩陣（FIM）**：衡量模型參數對任務的敏感度。透過估計 FIM 的對角線元素，將其作為任務的指紋（fingerprint）。
* **探測網路（Probe Network）**：使用一個**凍結**的預訓練模型（RoBERTa）作為骨幹，並在頂部訓練一個可學習的線性分類頭，僅計算這個分類頭的梯度資訊來生成任務向量。

這個方法能夠**量化**並**可視化**不同 NLP 任務之間的關係，有助於任務選擇、遷移學習和元學習（meta-learning）的研究。

---

## 程式碼檔案功能

* `data_cl_benchmark.py`：處理資料集載入與標籤轉換。
* `fim_text.py`：估計任務的 Fisher 資訊矩陣。
* `modeling_probe.py`：定義探測網路和分類模型。
* `run_embed.py`：主程式。

---

## 如何運行

### 1. 安裝套件
```bash
pip install torch transformers pandas scikit-learn numpy
```

### 2. 執行程式
使用以下指令在背景執行，並將日誌輸出到 `test.log`。
```bash
nohup python -m nlp_task2vec.run_embed \
  --data_root data/CL_Benchmark \
  --datasets "dbpedia,amazon,yahoo,agnews" \
  --backbone roberta-base \
  --epochs 5 --bs 32 --mc_steps 2 \
  --out_dir outputs > test.log 2>&1 &
```
* -\-datasets：指定要處理的任務清單。
* -\-epochs：訓練分類頭的次數（建議增加以提高準確度）。
* -\-out_dir：結果輸出資料夾。

---

## 輸出結果
程式執行完成後，`outputs/` 資料夾中將包含：

* `distance_matrix.csv`：所有任務兩兩之間的餘弦距離矩陣。
* `{task_name}_embed.npy`：每個任務的 Fisher 任務向量。
* `{task_name}_layers.json`：任務向量每個維度對應的層次名稱。