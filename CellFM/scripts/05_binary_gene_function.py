"""
05. Binary Gene Function Prediction (PyTorch)
데이터: datasets/CellFM/Gene_classification.h5ad (var 컬럼: t1/t2/t3, train_t1/t2/t3)
        datasets/CellFM/cellFM_embedding.pt (gene embedding)
출력:  results/05_binary_gene_function/
         metrics.json
         figures/acc_f1_bar.png
         figures/gene_embedding_umap.png
"""
import os, sys, datetime
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
FIG_DIR   = os.path.join(RESULTS, "figures")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
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


def save_figures(all_results, gene_emb, gene_index):
    logger.info("Figure 생성 시작...")

    # ── 1. ACC / F1 grouped bar chart ─────────────────────────────
    tasks = ["t1", "t2", "t3"]
    folds = [0, 1, 2, 3, 4]   # 5-fold
    x     = np.arange(len(tasks))
    width = 0.15
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#CCB974"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for fi, fold in enumerate(folds):
        accs = []
        f1s  = []
        for task in tasks:
            key = f"{task}_fold{fold}"
            res = all_results.get(key, {})
            accs.append(res.get("accuracy", 0))
            f1s.append(res.get("f1_macro", 0))
        offset = (fi - 2) * width
        ax1.bar(x + offset, accs, width, label=f"fold{fold}", color=colors[fi])
        ax2.bar(x + offset, f1s,  width, label=f"fold{fold}", color=colors[fi])

    for ax, title, ylabel in [
        (ax1, "Accuracy per Task/Fold", "Accuracy"),
        (ax2, "F1 (macro) per Task/Fold", "F1 Score"),
    ]:
        ax.set_xticks(x); ax.set_xticklabels(["T1", "T2", "T3"])
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "acc_f1_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("acc_f1_bar.png 저장")

    # ── 2. Gene Embedding UMAP ────────────────────────────────────
    try:
        adata_gene = sc.read_h5ad(DATA_PATH)
        common     = np.intersect1d(list(adata_gene.var_names), list(gene_index))
        adata_gene = adata_gene[:, common]
        geneset    = {j: i + 1 for i, j in enumerate(gene_index)}

        # t1 레이블이 있는 유전자만 사용
        gene_meta = adata_gene.var[adata_gene.var["train_t1"] > -1]
        gene_ids  = np.array([geneset[g] for g in gene_meta.index
                               if g in geneset], dtype=np.int32)
        t1_labels = gene_meta.loc[
            [g for g in gene_meta.index if g in geneset], "t1"
        ].values

        if len(gene_ids) == 0:
            raise ValueError("UMAP용 유전자 없음")

        emb_np = gene_emb.cpu().numpy()[gene_ids]   # [N, D]
        adata_emb = sc.AnnData(emb_np)
        adata_emb.obs["t1_label"] = t1_labels.astype(str)
        sc.pp.neighbors(adata_emb, use_rep="X", n_neighbors=15)
        sc.tl.umap(adata_emb, min_dist=0.3)
        fig = sc.pl.umap(adata_emb, color=["t1_label"], frameon=False,
                         title="Gene Embedding UMAP (T1 labels)",
                         return_fig=True, show=False)
        fig.savefig(os.path.join(FIG_DIR, "gene_embedding_umap.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("gene_embedding_umap.png 저장")
    except Exception as e:
        logger.warning(f"Gene Embedding UMAP 실패: {e}")


def main():
    gene_emb   = extract_gene_emb().to(DEVICE)
    gene_index = pd.read_csv(os.path.join(CSV_DIR, "gene_info.csv"),
                             index_col=0, header=0).index
    logger.info(f"gene_emb: {gene_emb.shape}, gene_index: {len(gene_index)}")

    all_results = {}
    for data_col in ["t1", "t2", "t3"]:
        for fold in [0, 1, 2, 3, 4]:   # 5-fold CV (값 범위: 0~4)
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

    try:
        save_figures(all_results, gene_emb, gene_index)
    except Exception as e:
        logger.error(f"Figure 생성 실패 (metrics는 이미 저장됨): {e}")


if __name__ == "__main__":
    main()
