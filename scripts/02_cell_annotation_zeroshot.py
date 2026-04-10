"""
02. Cell Annotation — Zero-shot (PyTorch + CellFM-torch)
backbone(extractor)을 동결하고 분류기 헤드만 학습.
데이터: datasets/CellFM/hPancreas_train.h5ad + hPancreas_test.h5ad
출력:  checkpoint/CellAnnotation/  (ckpt)
       results/02_cell_annotation_zeroshot/metrics.json
"""
import os, sys, datetime
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from scipy.sparse import csr_matrix as csm

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
PRETRAIN = os.path.join(ROOT_DIR, "checkpoint", "CellFM_80M_weight.ckpt")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

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
    codes = pd.Categorical(adata.obs["celltype"]).codes
    adata.obs["celltype_id"] = codes
    adata.obs["feat"]        = codes
    adata.obs["batch_id"]    = 0

    data = adata.X.astype(np.float32)
    T    = adata.X.sum(1)
    data = csm(np.round(data / np.maximum(1, T / 1e5, dtype=np.float32)))
    data.eliminate_zeros()
    adata.X = data

    n_cls = len(adata.obs["celltype"].unique())
    logger.info(f"데이터 로드: {adata.shape}, {n_cls} classes")
    return adata, n_cls


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
            with autocast():
                cls, _, _ = net(raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx)
            correct += (cls.argmax(1) == feat).sum().item()
            total   += len(feat)
    return correct / total if total > 0 else 0.0


def main():
    adata, n_cls = prepare_adata()

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
    prep_eval    = Prepare(cfg.nonz_len, pad=0, mask_ratio=0.0)   # 평가 시 마스킹 없음
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
    scaler    = GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    best_acc  = 0.0
    best_path = os.path.join(CKPT_DIR, "hPancreas_zeroshot_best.pth")

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
            with autocast():
                cls, mask_loss, _ = net(raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx)
                # zero-shot: cls loss만 (mask_loss는 frozen extractor에서 나오므로 backward skip)
                loss = criterion(cls, feat)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(net.cls.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            acc = (cls.argmax(1) == feat).float().mean().item()
            pbar.set_postfix(loss=total_loss/(step+1), acc=f"{acc:.3f}")

        scheduler.step()
        test_acc = eval_accuracy(net, test_loader, DEVICE)
        logger.info(f"Epoch {epoch+1}: cls_loss={total_loss/len(train_loader):.4f}  test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), best_path)
            logger.info(f"  ▶ 베스트 저장 acc={best_acc:.4f}")

    metrics = {
        "task":       TASK,
        "dataset":    "hPancreas",
        "epochs":     EPOCHS,
        "best_acc":   best_acc,
        "timestamp":  datetime.datetime.now().isoformat(),
        "checkpoint": best_path,
    }
    path = save_metrics(metrics, RESULTS)
    logger.info(f"결과 저장: {path}")


if __name__ == "__main__":
    main()
