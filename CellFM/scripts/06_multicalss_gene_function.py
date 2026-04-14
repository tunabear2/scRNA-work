"""
06. Multiclass Gene Function Prediction (GO term: MF / CC / BP)
데이터: datasets/CellFM/GO_data/{task}/top10_data/
        checkpoint/base_weight.ckpt → gene embedding 추출
출력:  checkpoint/GeneFunction/{task}_best_model.pth
       results/06_multiclass_gene_function/
         metrics.json
         figures/aupr_fmax_bar.png
         figures/pr_curve_{MF|CC|BP}.png
"""
import os, sys, datetime, json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from log_utils.logger import get_logger, save_metrics

TASK     = "06_multiclass_gene_function"
GO_DIR   = os.path.join(ROOT_DIR, "datasets", "CellFM", "GO_data")
CSV_DIR  = os.path.join(ROOT_DIR, "csv")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoint", "GeneFunction")
RESULTS  = os.path.join(ROOT_DIR, "results", TASK)
FIG_DIR  = os.path.join(RESULTS, "figures")
CKPT_BASE= os.path.join(ROOT_DIR, "checkpoint", "base_weight.ckpt")
EMB_PATH = os.path.join(ROOT_DIR, "datasets", "CellFM", "cellFM_embedding.pt")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

logger = get_logger(TASK, RESULTS)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, auc
import random, math


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_gene_emb():
    if os.path.exists(EMB_PATH):
        logger.info(f"기존 gene embedding 사용: {EMB_PATH}")
        return torch.load(EMB_PATH)

    logger.info("base_weight.ckpt에서 gene_emb 추출 중...")
    import mindspore as ms
    ckpt     = ms.load_checkpoint(CKPT_BASE)
    gene_emb = ckpt["gene_emb"].value().asnumpy()
    tensor   = torch.from_numpy(gene_emb)
    torch.save(tensor, EMB_PATH)
    logger.info(f"gene_emb 저장 완료: shape={tensor.shape}, path={EMB_PATH}")
    return tensor


class GOrna(Dataset):
    def __init__(self, data, label_map, baseline_index, model_type="cellfm"):
        common_gene = np.intersect1d(list(data.keys()), list(baseline_index))
        if model_type == "cellfm":
            self.geneset = {gene: idx + 1 for idx, gene in enumerate(baseline_index)}
        else:
            self.geneset = {gene: idx for idx, gene in enumerate(baseline_index)}
        self.gene       = np.array([self.geneset[g] for g in common_gene]).astype(np.int32)
        self.label_size = len(label_map)
        labels          = [set(data[g]) for g in common_gene]
        self.label      = [[label_map[item] for item in s] for s in labels]

    def __len__(self):  return len(self.gene)

    def __getitem__(self, idx):
        label_idx = torch.tensor(self.label[idx])
        label     = torch.zeros(self.label_size, dtype=torch.int)
        label[label_idx] = 1
        return torch.tensor(self.gene[idx]), label


class MLP_GO(nn.Module):
    def __init__(self, gene_emb, label_size, hidden_dim=1028, num_emb_layers=2, dropout=0.2):
        super().__init__()
        self.gene_emb   = gene_emb
        feature_dim     = gene_emb.shape[-1]
        self.input_block= nn.Sequential(
            nn.LayerNorm(feature_dim, eps=1e-6),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
        )
        hidden_layers = []
        for i in range(num_emb_layers - 1):
            hidden_layers += [
                nn.LayerNorm(hidden_dim, eps=1e-6),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
            if i == num_emb_layers - 2:
                hidden_layers.append(nn.LayerNorm(hidden_dim, eps=1e-6))
        self.hidden_block = nn.Sequential(*hidden_layers)
        self.output_block = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.Dropout(0.2), nn.SiLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2), nn.SiLU(),
            nn.Linear(256, label_size),
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, gene_id):
        h = self.gene_emb[gene_id].to(torch.float32)
        h = self.input_block(h)
        h = self.output_block(h)
        return h


def run_go_task(task, top, gene_emb, model_idx, seed=5, num_epochs=5, batch=4, interval=0.01):
    """task: 'MF'|'CC'|'BP'
    Returns dict with AUPR, Fmax, best_epoch, checkpoint,
    and pr_curve (prec, rec arrays) for plotting.
    """
    set_seed(seed)
    device = "cuda:0"
    logger.info(f"── GO 태스크: {task} top{top} ──")

    data_dir = os.path.join(GO_DIR, task, f"top{top}_data")
    df_train = pd.read_csv(os.path.join(data_dir, "processed_train.csv"))
    df_valid = pd.read_csv(os.path.join(data_dir, "processed_valid.csv"))
    df_test  = pd.read_csv(os.path.join(data_dir, "processed_test.csv"))
    with open(os.path.join(data_dir, "func_dict.json")) as f:
        label_dict = json.load(f)
    label_size = len(label_dict)

    key        = "gene"
    data_train = df_train.groupby(key)["go"].apply(list).to_dict()
    data_valid = df_valid.groupby(key)["go"].apply(list).to_dict()
    data_test  = df_test.groupby(key)["go"].apply(list).to_dict()

    gene_emb_dev = gene_emb.to(device)
    train_set    = GOrna(data_train, label_dict, model_idx)
    valid_set    = GOrna(data_valid, label_dict, model_idx)
    test_set     = GOrna(data_test,  label_dict, model_idx)
    logger.info(f"샘플 수: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1024, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=1024, shuffle=False)

    model     = MLP_GO(gene_emb_dev, label_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

    best_valid_loss = float("inf")
    best_model_path = os.path.join(CKPT_DIR, f"{task}_top{top}_best.pth")
    best_epoch      = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, all_labels, all_scores = 0, [], []
        for batch_ids, batch_labels in train_loader:
            batch_ids, batch_labels = batch_ids.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_ids)
            loss    = criterion(outputs, batch_labels.float())
            pos_w   = batch_labels.sum()
            neg_w   = (1 - batch_labels).sum()
            loss    = (loss * batch_labels).sum() / pos_w + \
                      (loss * (1 - batch_labels)).sum() / neg_w
            if math.isnan(loss.item()):
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_labels.append(batch_labels.cpu().numpy())
            all_scores.append(torch.sigmoid(outputs).cpu().detach().numpy())

        train_aupr = average_precision_score(
            np.concatenate(all_labels), np.concatenate(all_scores))

        model.eval()
        val_loss, y_true, y_scores = 0, [], []
        with torch.no_grad():
            for batch_ids, batch_labels in valid_loader:
                batch_ids, batch_labels = batch_ids.to(device), batch_labels.to(device)
                outputs  = model(batch_ids)
                loss     = criterion(outputs, batch_labels.float())
                pos_w    = batch_labels.sum()
                neg_w    = (1 - batch_labels).sum()
                loss     = (loss * batch_labels).sum() / pos_w + \
                           (loss * (1 - batch_labels)).sum() / neg_w
                val_loss += loss.item()
                y_true.append(batch_labels.cpu().numpy())
                y_scores.append(torch.sigmoid(outputs).cpu().numpy())

        y_true   = np.concatenate(y_true).reshape(-1)
        y_scores = np.concatenate(y_scores).reshape(-1)
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        aupr     = auc(rec, prec)
        avg_val  = val_loss / len(valid_loader)

        best_f1  = 0
        for thr in np.arange(0.0, 1.0, interval):
            preds   = (y_scores >= thr).astype(int)
            best_f1 = max(best_f1, f1_score(y_true, preds, average="macro"))

        logger.info(f"Epoch [{epoch+1}/{num_epochs}] "
                    f"train_loss={total_loss/len(train_loader):.4f} "
                    f"train_AUPR={train_aupr:.4f} "
                    f"val_loss={avg_val:.4f} val_AUPR={aupr:.4f} Fmax={best_f1:.4f}")

        if avg_val < best_valid_loss:
            best_valid_loss = avg_val
            best_epoch      = epoch
            torch.save(model.state_dict(), best_model_path)

    # ── 테스트 ──────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch_ids, batch_labels in test_loader:
            batch_ids, batch_labels = batch_ids.to(device), batch_labels.to(device)
            outputs  = model(batch_ids)
            y_true.append(batch_labels.cpu().numpy())
            y_scores.append(torch.sigmoid(outputs).cpu().numpy())

    y_true   = np.concatenate(y_true).reshape(-1)
    y_scores = np.concatenate(y_scores).reshape(-1)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    test_aupr = auc(rec, prec)
    test_fmax = 0
    for thr in np.arange(0.0, 1.0, interval):
        preds     = (y_scores >= thr).astype(int)
        test_fmax = max(test_fmax, f1_score(y_true, preds, average="macro"))

    logger.info(f"{task} 테스트 결과: best_epoch={best_epoch}, "
                f"AUPR={test_aupr:.4f}, Fmax={test_fmax:.4f}")
    return {
        "AUPR":       test_aupr,
        "Fmax":       test_fmax,
        "best_epoch": best_epoch,
        "checkpoint": best_model_path,
        "_pr_prec":   prec.tolist(),   # PR curve 저장 (figure용)
        "_pr_rec":    rec.tolist(),
    }


def save_figures(all_results):
    logger.info("Figure 생성 시작...")
    tasks = [t for t in ["MF", "CC", "BP"] if t in all_results and "AUPR" in all_results[t]]

    if not tasks:
        logger.warning("유효한 태스크 결과 없음, figure 생성 건너뜀")
        return

    # ── 1. AUPR / Fmax bar chart ────────────────────────────────
    auprs = [all_results[t]["AUPR"] for t in tasks]
    fmaxs = [all_results[t]["Fmax"] for t in tasks]
    x     = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, auprs, width, label="AUPR",  color="#4C72B0")
    ax.bar(x + width/2, fmaxs, width, label="Fmax",  color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("Score"); ax.set_title("GO Function Prediction (MF / CC / BP)")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (a, f) in enumerate(zip(auprs, fmaxs)):
        ax.text(i - width/2, a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, f + 0.01, f"{f:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "aupr_fmax_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("aupr_fmax_bar.png 저장")

    # ── 2. PR curves ────────────────────────────────────────────
    colors = {"MF": "#4C72B0", "CC": "#DD8452", "BP": "#55A868"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for t in tasks:
        prec = np.array(all_results[t].get("_pr_prec", []))
        rec  = np.array(all_results[t].get("_pr_rec",  []))
        if len(prec) == 0:
            continue
        aupr = all_results[t]["AUPR"]
        ax.plot(rec, prec, label=f"{t} (AUPR={aupr:.3f})", color=colors.get(t))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("pr_curves.png 저장")


def main():
    gene_emb  = extract_gene_emb()
    model_idx = pd.read_csv(os.path.join(CSV_DIR, "gene_info.csv"))["HGNC_gene"]
    logger.info(f"gene_emb: {gene_emb.shape}, gene_index: {len(model_idx)}")

    all_results = {}
    for task in ["MF", "CC", "BP"]:
        try:
            res = run_go_task(task, top=10, gene_emb=gene_emb, model_idx=model_idx)
            all_results[task] = res
        except Exception as e:
            logger.error(f"{task} 실패: {e}")
            all_results[task] = {"error": str(e)}

    logger.info(f"전체 결과: {all_results}")

    # metrics에서 내부 PR curve 배열은 제외
    metrics_clean = {}
    for task, res in all_results.items():
        metrics_clean[task] = {k: v for k, v in res.items() if not k.startswith("_")}

    metrics = {
        "task":      TASK,
        "dataset":   "GO_data (MF/CC/BP top10)",
        "results":   metrics_clean,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    path = save_metrics(metrics, RESULTS)
    logger.info(f"결과 저장: {path}")

    try:
        save_figures(all_results)
    except Exception as e:
        logger.error(f"Figure 생성 실패 (metrics는 이미 저장됨): {e}")


if __name__ == "__main__":
    main()
