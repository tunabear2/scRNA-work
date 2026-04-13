"""
02. Cell Annotation — Zero-shot (PyTorch + CellFM-torch)
backbone(extractor)을 동결하고 분류기 헤드만 학습.
데이터: datasets/CellFM/hPancreas_train.h5ad + hPancreas_test.h5ad
출력:  checkpoint/CellAnnotation/hPancreas_zeroshot_best.pth
       results/02_cell_annotation_zeroshot/
         metrics.json, predicted_labels.csv
         figures/learning_curve.png
         figures/confusion_matrix.png
         figures/umap_true_label.png
         figures/umap_pred_label.png
"""
import os, sys, datetime
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from scipy.sparse import csr_matrix as csm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR   = os.path.join(ROOT_DIR, "scripts")
CELLFM_TORCH = os.path.join(ROOT_DIR, "CellFM-torch")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CELLFM_TORCH)

os.chdir(ROOT_DIR)

from log_utils.logger import get_logger, save_metrics
from layers.utils import Config_80M, SCrna, Prepare, build_dataset
from model import Finetune_Cell_FM

TASK     = "02_cell_annotation_zeroshot"
DATA_DIR = os.path.join(ROOT_DIR, "datasets", "CellFM")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoint", "CellAnnotation")
RESULTS  = os.path.join(ROOT_DIR, "results", TASK)
FIG_DIR  = os.path.join(RESULTS, "figures")
PRETRAIN = os.path.join(ROOT_DIR, "checkpoint", "CellFM_80M_weight.ckpt")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

logger = get_logger(TASK, RESULTS)

DEVICE = "cuda:0"
EPOCHS = 30
BATCH  = 16


def prepare_adata():
    train_data = sc.read_h5ad(os.path.join(DATA_DIR, "hPancreas_train.h5ad"))
    test_data  = sc.read_h5ad(os.path.join(DATA_DIR, "hPancreas_test.h5ad"))

    for d in (train_data, test_data):
        if "Celltype" in d.obs.columns:
            d.obs["celltype"] = d.obs["Celltype"].astype(str)
        elif "cell_type" in d.obs.columns:
            d.obs["celltype"] = d.obs["cell_type"].astype(str)

    train_data.obs["train"] = 0
    test_data.obs["train"]  = 2
    adata = ad.concat([train_data, test_data], join="outer")

    adata.obs["celltype"] = adata.obs["celltype"].astype(str)
    cat      = pd.Categorical(adata.obs["celltype"])
    codes    = cat.codes
    id2label = dict(enumerate(cat.categories))

    adata.obs["celltype_id"] = codes
    adata.obs["feat"]        = codes
    adata.obs["batch_id"]    = 0

    data = adata.X.astype(np.float32)
    T    = adata.X.sum(1)
    data = csm(np.round(data / np.maximum(1, T / 1e5, dtype=np.float32)))
    data.eliminate_zeros()
    adata.X = data

    n_cls = len(id2label)
    logger.info(f"데이터 로드: {adata.shape}, {n_cls} classes")
    return adata, n_cls, id2label


def eval_accuracy(net, loader, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            raw_nzdata = batch["raw_nzdata"].to(device)
            dw_nzdata  = batch["dw_nzdata"].to(device)
            ST_feat    = batch["ST_feat"].to(device)
            nonz_gene  = batch["nonz_gene"].to(device)
            mask_gene  = batch["mask_gene"].to(device)
            zero_idx   = batch["zero_idx"].to(device)
            feat       = batch["feat"].long().to(device)
            with autocast("cuda"):
                cls, _, _ = net(raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx)
            correct += (cls.argmax(1) == feat).sum().item()
            total   += len(feat)
    return correct / total if total > 0 else 0.0


def collect_test_results(net, loader, device):
    """테스트셋 예측값, cls_token 임베딩, 레이블 수집."""
    net.eval()
    all_preds, all_labels, all_embs = [], [], []
    with torch.no_grad():
        for batch in loader:
            raw_nzdata = batch["raw_nzdata"].to(device)
            dw_nzdata  = batch["dw_nzdata"].to(device)
            ST_feat    = batch["ST_feat"].to(device)
            nonz_gene  = batch["nonz_gene"].to(device)
            mask_gene  = batch["mask_gene"].to(device)
            zero_idx   = batch["zero_idx"].to(device)
            feat       = batch["feat"].long().to(device)
            with autocast("cuda"):
                cls, _, emb = net(raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx)
            all_preds.extend(cls.argmax(1).cpu().numpy())
            all_labels.extend(feat.cpu().numpy())
            all_embs.append(emb.cpu().float().numpy())
    return np.array(all_preds), np.array(all_labels), np.concatenate(all_embs)


def save_figures(net, test_loader, id2label, train_losses, test_accs):
    logger.info("Figure 생성 시작...")

    # ── 1. 학습 곡선 ──────────────────────────────────────────────
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses, marker="o", ms=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cls Loss"); ax1.set_title("Training Loss (cls only)")
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, test_accs, marker="o", ms=3, color="orange")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Test Accuracy"); ax2.set_title("Test Accuracy")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "learning_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("learning_curve.png 저장")

    # ── 2. 테스트 예측 + 임베딩 수집 ─────────────────────────────
    all_preds, all_labels, all_embs = collect_test_results(net, test_loader, DEVICE)

    # 예측 레이블 CSV
    df_pred = pd.DataFrame({
        "true_label": [id2label[i] for i in all_labels],
        "pred_label": [id2label[i] for i in all_preds],
        "correct":    (all_preds == all_labels).astype(int),
    })
    df_pred.to_csv(os.path.join(RESULTS, "predicted_labels.csv"), index=False)
    logger.info("predicted_labels.csv 저장")

    # ── 3. Confusion matrix ────────────────────────────────────────
    label_names = [id2label[i] for i in sorted(id2label)]
    cm  = confusion_matrix(all_labels, all_preds)
    n   = len(label_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix (Test Set, Zero-shot)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("confusion_matrix.png 저장")

    # ── 4. UMAP (true label / pred label) ─────────────────────────
    adata_test = sc.AnnData(all_embs)
    adata_test.obs["true_label"] = [id2label[i] for i in all_labels]
    adata_test.obs["pred_label"] = [id2label[i] for i in all_preds]
    sc.pp.neighbors(adata_test, use_rep="X", n_neighbors=15)
    sc.tl.umap(adata_test, min_dist=0.3)
    for color_key in ["true_label", "pred_label"]:
        fig = sc.pl.umap(adata_test, color=[color_key], frameon=False,
                         return_fig=True, show=False)
        fig.savefig(os.path.join(FIG_DIR, f"umap_{color_key}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"umap_{color_key}.png 저장")


def main():
    adata, n_cls, id2label = prepare_adata()

    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs       = True
    cfg.add_zero  = True
    cfg.pad_zero  = True
    cfg.num_cls   = n_cls
    cfg.mask_ratio= 0.5
    cfg.ckpt_path = PRETRAIN
    cfg.device    = DEVICE

    trainset = SCrna(adata, mode="train")
    testset  = SCrna(adata, mode="test")
    logger.info(f"train={len(trainset)}, test={len(testset)}")

    prep         = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
    prep_eval    = Prepare(cfg.nonz_len, pad=0, mask_ratio=0.0)
    train_loader = build_dataset(trainset, prep,      batch_size=BATCH, pad_zero=True, drop=True,  shuffle=True)
    test_loader  = build_dataset(testset,  prep_eval, batch_size=BATCH, pad_zero=True, drop=False, shuffle=False)

    net = Finetune_Cell_FM(cfg).to(DEVICE)
    net.extractor.load_model(weight=True, moment=False)

    # ── Zero-shot: backbone 동결, cls 헤드만 학습 ──────────────────
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("cls.")
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f"Backbone 동결 완료. 학습 가능 파라미터: {trainable:,}")

    optimizer = AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-5)
    scaler    = GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    best_acc  = 0.0
    best_path = os.path.join(CKPT_DIR, "hPancreas_zeroshot_best.pth")
    train_losses, test_accs = [], []

    for epoch in range(EPOCHS):
        net.train()
        # extractor는 항상 eval 모드로 유지 (BN/dropout 동결)
        net.extractor.eval()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[02 Zeroshot] Ep {epoch+1}/{EPOCHS}")
        for step, batch in enumerate(pbar):
            raw_nzdata = batch["raw_nzdata"].to(DEVICE)
            dw_nzdata  = batch["dw_nzdata"].to(DEVICE)
            ST_feat    = batch["ST_feat"].to(DEVICE)
            nonz_gene  = batch["nonz_gene"].to(DEVICE)
            mask_gene  = batch["mask_gene"].to(DEVICE)
            zero_idx   = batch["zero_idx"].to(DEVICE)
            feat       = batch["feat"].long().to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                cls, mask_loss, _ = net(raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx)
                # zero-shot: cls loss만 (frozen extractor)
                loss = criterion(cls, feat)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(net.cls.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            acc = (cls.argmax(1) == feat).float().mean().item()
            pbar.set_postfix(loss=total_loss/(step+1), acc=f"{acc:.3f}")

        scheduler.step()
        epoch_loss = total_loss / len(train_loader)
        test_acc   = eval_accuracy(net, test_loader, DEVICE)
        train_losses.append(epoch_loss)
        test_accs.append(test_acc)
        logger.info(f"Epoch {epoch+1}: cls_loss={epoch_loss:.4f}  test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), best_path)
            logger.info(f"  ▶ 베스트 저장 acc={best_acc:.4f}")

    # 베스트 모델 로드 후 figure 생성
    net.load_state_dict(torch.load(best_path))
    save_figures(net, test_loader, id2label, train_losses, test_accs)

    metrics = {
        "task":         TASK,
        "dataset":      "hPancreas",
        "epochs":       EPOCHS,
        "best_acc":     best_acc,
        "train_losses": train_losses,
        "test_accs":    test_accs,
        "timestamp":    datetime.datetime.now().isoformat(),
        "checkpoint":   best_path,
    }
    path = save_metrics(metrics, RESULTS)
    logger.info(f"결과 저장: {path}")


if __name__ == "__main__":
    main()
