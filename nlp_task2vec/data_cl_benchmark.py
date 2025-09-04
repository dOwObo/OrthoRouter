# nlp_task2vec/data_cl_benchmark.py
import os, json, glob
import pandas as pd

def _read_json(path):
    """
    只讀取 JSON 檔案，返回 DataFrame
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    elif isinstance(obj, dict):
        return pd.DataFrame([obj])
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

def _pick_col(df, cands):
    """
    選擇存在的欄位，若無則回傳第一欄
    """
    for c in cands:
        if c in df.columns:
            return c
    return df.columns[0]

def load_labels(dir_path):
    """
    讀取 labels.json 並建立 label -> id 映射
    """
    labels_path = os.path.join(dir_path, "labels.json")
    if not os.path.exists(labels_path):
        return None
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_list = json.load(f)
    label2id = {label: idx for idx, label in enumerate(labels_list)}
    return label2id

def load_split(dir_path):
    """
    讀取 train/dev/test 並轉換 label 為整數
    """
    label2id = load_labels(dir_path)
    id2label = {v: k for k, v in label2id.items()} if label2id else None

    # 找 train/dev/test 檔案
    def find_split(name):
        pats = [f"*{name}.*", f"*{name.upper()}.*", f"*{name.lower()}.*"]
        files = []
        for p in pats:
            files += glob.glob(os.path.join(dir_path, p))
        return files[0] if files else None

    train_p = find_split("train")
    dev_p   = find_split("dev")
    test_p  = find_split("test")

    def load_df(p):
        if not p:
            return None
        df = _read_json(p)
        df = df[["sentence", "label"]].dropna()

        # 轉換 label 為整數
        if label2id is not None:
            df["label"] = df["label"].map(label2id)
        else:
            if df["label"].dtype == object:
                mapping = {v: i for i, v in enumerate(sorted(df["label"].unique()))}
                df["label"] = df["label"].map(mapping)
        return df

    return load_df(train_p), load_df(dev_p), load_df(test_p), id2label

def dataset_roots(root):
    """
    對應 CL_Benchmark 資料夾的路徑
    """
    return {
        "agnews": os.path.join(root, "TC", "agnews"),
        "dbpedia": os.path.join(root, "TC", "dbpedia"),
        "yahoo": os.path.join(root, "TC", "yahoo"),
        "amazon": os.path.join(root, "SC", "amazon"),
    }

# 範例使用
if __name__ == "__main__":
    root = "./CL_Benchmark"
    roots = dataset_roots(root)
    for name, path in roots.items():
        train, dev, test = load_split(path)
        print(f"{name}: train={len(train)}, dev={len(dev) if dev is not None else 0}, test={len(test) if test is not None else 0}")
