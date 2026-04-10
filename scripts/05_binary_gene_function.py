"""
05. Binary Gene Function Prediction (PyTorch)
데이터: datasets/CellFM/Gene_classification.h5ad (var 컬럼: t1/t2/t3, train_t1/t2/t3)
        datasets/CellFM/cellFM_embedding.pt (gene embedding)
출력:  results/05_binary_gene_function/metrics.json
"""
import os, sys, datetime
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from log_utils.logger import get_logger, save_metrics

TASK      = "05_binary_gene_function"
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "CellFM", "Gene_classification.h5ad")
CSV_DIR   = os.path.join(ROOT_DIR, "csv")
CKPT_BASE = os.path.join(ROOT_DIR, "checkpoint", "base_weight.ckpt")
EMB_PATH  = os.path.join(ROOT_DIR, "datasets", "CellFM", "cellFM_embedding.pt")
RESULTS   = os.path.join(ROOT_DIR, "results", TASK)

os.makedirs(RESULTS, exist_ok=True)
logger = get_logger(TASK, RESULTS)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

DEVICE = "cuda:0"
EPOCHS = 30


def extract_gene_emb():
    if os.path.exists(EMB_PATH):
        logger.info(f"기존 gene embedding 사용: {EMB_PATH}")
        return torch.load(EMB_PATH)
    logger.info("base_weight.ckpt에서 gene_emb 추출 중 (MindSpore CPU)...")
    import mindspore as ms
    ckpt     = ms.load_checkpoint(CKPT_BASE)
    gene_emb = ckpt["gene_emb"].value().asnumpy()
    tensor   = torch.from_numpy(gene_emb)
    torch.save(tensor, EMB_PATH)
    logger.info(f"gene_emb 저장: shape={tensor.shape}, path={EMB_PATH}")
    return tensor


class GeneDataset(Dataset):
    def __init__(self, gene_ids, labels):
        self.gene_ids = torch.tensor(gene_ids, dtype=torch.long)
        self.labels   = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.gene_ids)

    def __getitem__(self, idx):
        return self.gene_ids[idx], self.labels[idx]


class MLP(nn.Module):
    def __init__(self, gene_emb):
        super().__init__()
        d = gene_emb.shape[-1]
        # gene_emb은 frozen
        self.register_buffer("gene_emb", gene_emb.float())
        self.net = nn.Sequential(
            nn.Linear(d, d // 2, bias=False),
            nn.Dropout(0.15),
            nn.SiLU(),
            nn.Linear(d // 2, d // 4, bias=False),
            nn.Dropout(0.15),
            nn.SiLU(),
            nn.Linear(d // 4, 2, bias=False),
        )

    def forward(self, gene_id):
        x = self.gene_emb[gene_id]
        return self.net(x)


def run_task(data_col, fold, gene_index, gene_emb):
    logger.info(f"── 태스크: {data_col}, fold={fold} ──")
    adata = sc.read_h5ad(DATA_PATH)

    common = np.intersect1d(list(adata.var_names), list(gene_index))
    adata  = adata[:, common]
    geneset = {j: i + 1 for i, j in enumerate(gene_index)}

    gene_meta = adata.var[adata.var[f"train_{data_col}"] > -1]
    idx_test  = gene_meta[f"train_{data_col}"] == fold

    train_genes = np.array([geneset[g] for g in gene_meta[~idx_test].index], dtype=np.int32)
    train_labels= gene_meta[data_col][~idx_test].values.astype(np.int64)
    test_genes  = np.array([geneset[g] for g in gene_meta[idx_test].index],  dtype=np.int32)
    test_labels = gene_meta[data_col][idx_test].values.astype(np.int64)

    logger.info(f"  train={len(train_genes)}, test={len(test_genes)}")
    if len(test_genes) == 0:
        return {"error": "no test samples"}

    train_ds = GeneDataset(train_genes, train_labels)
    test_ds  = GeneDataset(test_genes,  test_labels)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, drop_last=False)

    model     = MLP(gene_emb).to(DEVICE)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for gene_id, label in train_loader:
            gene_id, label = gene_id.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out  = model(gene_id)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for gene_id, label in test_loader:
            gene_id = gene_id.to(DEVICE)
            out     = model(gene_id)
            preds   = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    logger.info(f"  결과: acc={acc:.4f}, F1={f1:.4f}")
    return {"accuracy": acc, "f1_macro": f1}


def main():
    gene_emb  = extract_gene_emb().to(DEVICE)
    gene_index = pd.read_csv(os.path.join(CSV_DIR, "gene_info.csv"),
                             index_col=0, header=0).index
    logger.info(f"gene_emb: {gene_emb.shape}, gene_index: {len(gene_index)}")

    all_results = {}
    for data_col in ["t1", "t2", "t3"]:
        for fold in [1, 2, 3]:
            key = f"{data_col}_fold{fold}"
            try:
                res = run_task(data_col, fold, gene_index, gene_emb)
                all_results[key] = res
            except Exception as e:
                logger.error(f"{key} 실패: {e}")
                all_results[key] = {"error": str(e)}

    logger.info(f"전체 결과: {all_results}")
    metrics = {
        "task":      TASK,
        "dataset":   "Gene_classification",
        "epochs":    EPOCHS,
        "results":   all_results,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    path = save_metrics(metrics, RESULTS)
    logger.info(f"결과 저장: {path}")


if __name__ == "__main__":
    main()
