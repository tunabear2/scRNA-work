#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT: Fine-tuning Pre-trained Model for Multiomic Integration
Converted from Tutorial_Multiomics__1_ Jupyter Notebook.

Pipeline:
  1. 하이퍼파라미터 설정
  2. 데이터 로드 및 전처리 (RNA + Protein)
  3. 사전학습 모델 로드 및 임베딩 확장
  4. Task-specific 목표로 파인튜닝
  5. 평가

Dataset: BMMC (CITE-seq: RNA + Protein, 다중 배치)
Pre-trained model: scGPT_human (whole-body)

Environment Requirements:
  anndata==0.8.0      scanpy==1.9.1       torch==1.13.0
  numpy==1.21.6       pandas==1.3.5       scipy==1.7.3
  matplotlib==3.5.2   seaborn==0.11.2     scikit-learn==1.0.2
  umap-learn==0.5.3   scvi-tools==0.16.4  leidenalg==0.8.10
"""

import copy
import gc
import json
import os
import sys
import time
import traceback
import types
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경용 non-interactive 백엔드
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch
from anndata import AnnData
from scipy.sparse import issparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab

sys.path.insert(0, "../")

import scgpt as scg

# scgpt 라이브러리에서 공용 함수 임포트
# (define_wandb_metrcis는 라이브러리 내 오타이므로 그대로 사용)
from scgpt import (
    define_wandb_metrcis,
    eval_testdata,
    evaluate,
    prepare_data,
    prepare_dataloader,
    train,
)
from scgpt.loss import (
    criterion_neg_log_bernoulli,
    masked_mse_loss,
    masked_relative_error,
)
from scgpt.model import MultiOmicTransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import random_mask_value, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import category_str2int, eval_scib_metrics, set_seed

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")


# ======================================================================
# wandb 심(Shim): 로컬 실행 시 W&B 없이도 동작하도록 대체 구현
# 실제 W&B 추적이 필요하다면 `pip install wandb` 후 로그인하면 자동으로 실제 wandb 사용
# ======================================================================

class _WandbShim:
    """W&B 미설치 / 미로그인 환경을 위한 최소 호환 shim."""

    class Settings:
        def __init__(self, **kwargs):
            pass

    class Artifact:
        def __init__(self, name, type=None):
            self.name = name

        def add_file(self, path):
            pass

    class Image:
        def __init__(self, path, caption=None):
            pass

    class _Run:
        config = None

        def log(self, d, **kw):
            pass

        def log_artifact(self, a):
            pass

        def finish(self):
            pass

    @staticmethod
    def init(config=None, project=None, reinit=False, settings=None):
        run = _WandbShim._Run()
        run.config = (
            types.SimpleNamespace(**config)
            if isinstance(config, dict)
            else config
        )
        return run

    @staticmethod
    def watch(model, **kwargs):
        pass

    @staticmethod
    def define_metric(name, **kwargs):
        pass

    @staticmethod
    def log(d, **kw):
        pass

    @staticmethod
    def finish():
        pass


try:
    import wandb as _wandb_real
    _wandb_real.login()  # 미로그인 시 예외 발생 → shim으로 폴백
    wandb = _wandb_real
except Exception:
    wandb = _WandbShim()


# ======================================================================
# 1. 하이퍼파라미터 설정 (Step 1: Hyper-parameter Setup)
# RNA + Protein 멀티오믹 통합을 위한 설정.
# 배치 보정을 위해 DAR 목표를 활성화하고,
# 모달리티 인식 학습을 위해 use_mod=True 사용.
# ======================================================================

hyperparameter_defaults = dict(
    task="multiomic",
    seed=42,
    dataset_name="BMMC",
    do_train=True,
    load_model="../save/scGPT_human",
    freeze=False,
    GEP=True,           # Gene Expression Prediction
    GEPC=True,          # Gene Expression Prediction for Cell objective
    CLS=False,          # Cell type classification
    ESC=False,          # Elastic cell similarity
    DAR=True,           # Domain Adversarial Regularization (배치 보정)
    DSBN=False,         # Domain-specific batch normalization
    mask_ratio=0.4,
    explicit_zero_prob=False,
    ecs_thres=0,        # 0.0 = 비활성화
    dab_weight=1.0,
    use_batch_labels=True,
    use_mod=True,       # 모달리티 인식 학습
    per_seq_batch_sample=False,
    epochs=25,
    input_layer_key="X_binned",
    n_bins=51,
    n_hvg=1200,
    n_hvp=4000,
    max_seq_len=4001,   # n_hvg + n_hvp + 1 (<cls> 토큰)
    lr=1e-3,
    batch_size=16,
    layer_size=512,
    nlayers=4,
    nhead=8,            # 사전학습 모델 로드 시 args.json으로 덮어씌워짐
    dropout=0.2,
    schedule_ratio=0.95,
    save_eval_interval=5,
    log_interval=100,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    pad_token="<pad>",
    mask_value=-1,
    pad_value=-2,
    include_zero_gene=False,
)

run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = run.config if isinstance(wandb, _WandbShim) else wandb.config
print(config)

set_seed(config.seed)

special_tokens = [config.pad_token, "<cls>", "<eoc>"]


# ======================================================================
# 2. 저장 경로 및 로거 초기화
# ======================================================================

dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving to {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


# ======================================================================
# 3. 데이터 로드 (Step 2.1: Load BMMC Data)
# 데이터 출처: https://drive.google.com/file/d/10RxboePS5p2Jj2Sfq1Ghzgqnl6nqPv5V/
# 원본 90K+ 세포 중 첫 3 donor (배치)의 B/Mono/T 세포 서브타입 12K+ 서브셋 사용
# ======================================================================

if dataset_name == "BMMC":
    adata = sc.read("BMMC_processed.h5ad")

    # 첫 3개 donor와 17개 세포 타입 서브셋
    donor_mask = adata.obs.DonorID.isin([10886, 11466, 12710])
    celltype_mask = adata.obs.cell_type.isin(
        np.unique(adata.obs.cell_type.values)[:17]
    )
    adata = adata[donor_mask & celltype_mask]

    adata.obs["celltype"] = adata.obs["cell_type"].astype(str).astype("category")
    adata.var["gene_name"] = adata.var.index.tolist()

    # 배치 레이블 인코딩
    le = preprocessing.LabelEncoder()
    adata.obs["batch_id"] = le.fit_transform(adata.obs["batch"].values)
    adata.obs["str_batch"] = adata.obs["batch_id"].astype("category")

    # RNA / Protein 모달리티 분리
    adata_protein = adata[:, adata.var.feature_types.isin(["ADT"])].copy()
    adata_protein.var.index = ["p_" + i for i in adata_protein.var.index]
    adata = adata[:, adata.var.feature_types.isin(["GEX"])].copy()
    data_is_raw = False

# 모달리티 메타데이터 구성
if config.use_mod:
    gene_rna_df = pd.DataFrame({"mod": "RNA"}, index=adata.var.index.tolist())
    gene_protein_df = pd.DataFrame(
        {"mod": "Protein"}, index=adata_protein.var.index.tolist()
    )
    gene_loc_df = pd.concat([gene_rna_df, gene_protein_df])
    gene_loc_df["mod"] = gene_loc_df["mod"].astype("category")


# ======================================================================
# 4. 어휘(Vocab) 및 유전자 ID 설정 (Step 2.2: Cross-check Gene Set)
# 사전학습 모델과 데이터 간 공통 유전자만 유지,
# 새로운 Protein 토큰은 임베딩 레이어에 추가
# ======================================================================

if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1
        for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"Match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    old_vocab = vocab

# 모델 차원 설정 (사전학습 모델 없을 때 config 값 사용)
embsize = config.layer_size
nhead = config.nhead
nlayers = config.nlayers
d_hid = config.layer_size


# ======================================================================
# 5. 데이터 전처리 (Step 2.3: Pre-process)
# ======================================================================

# 5-1. RNA 전처리: 정규화 → log1p → HVG 선택 → 값 비닝
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=1,
    filter_cell_by_counts=1,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=data_is_raw,
    result_log1p_key="X_log1p",
    subset_hvg=config.n_hvg,
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,
    result_binned_key="X_binned",
)
preprocessor(adata, batch_key=None)

# 5-2. Protein 전처리: 정규화 생략, 비닝만 수행
preprocessor_protein = Preprocessor(
    use_key="X",
    filter_gene_by_counts=0,
    filter_cell_by_counts=False,
    normalize_total=False,
    result_normed_key="X_normed",
    log1p=False,
    result_log1p_key="X_log1p",
    subset_hvg=False,
    hvg_flavor=None,
    binning=config.n_bins,
    result_binned_key="X_binned",
)
preprocessor_protein(adata_protein, batch_key=None)

# 5-3. RNA + Protein 결합
data_combined = np.concatenate(
    [adata.layers["X_binned"], adata_protein.layers["X_binned"]], axis=1
)
adata = AnnData(
    X=data_combined,
    obs=adata.obs,
    var=pd.DataFrame(
        index=adata.var_names.tolist() + adata_protein.var_names.tolist()
    ),
    layers={"X_binned": data_combined},
)
adata.var["gene_name"] = adata.var.index.tolist()

if config.per_seq_batch_sample:
    adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()


# ======================================================================
# 6. 입력 데이터 토크나이즈 (Step 2.4: Tokenize)
# ======================================================================

all_counts = (
    adata.layers[config.input_layer_key].toarray()
    if issparse(adata.layers[config.input_layer_key])
    else adata.layers[config.input_layer_key]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels = np.array(adata.obs["celltype"].tolist())
num_types = len(set(celltypes_labels))

batch_ids = np.array(adata.obs["batch_id"].tolist())
num_batch_types = len(set(batch_ids))

# 모달리티 타입 인코딩
if config.use_mod:
    mod_type = np.array([gene_loc_df.loc[g, "mod"] for g in genes])
    vocab_mod = Vocab(
        VocabPybind(np.unique(gene_loc_df["mod"]).tolist() + special_tokens, None)
    )
    vocab_mod.set_default_index(vocab_mod["<pad>"])
    mod_type = np.array(vocab_mod(list(mod_type)), dtype=int)
    ntokens_mod = len(vocab_mod)

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

# 학습 데이터 통계 출력
num_of_non_zero_genes = [
    np.count_nonzero(train_data[i]) for i in range(train_data.shape[0])
]
logger.info(f"Max non-zero genes per cell    : {np.max(num_of_non_zero_genes)}")
logger.info(f"Min non-zero genes per cell    : {np.min(num_of_non_zero_genes)}")
logger.info(f"Avg non-zero genes per cell    : {np.mean(num_of_non_zero_genes):.2f}")
logger.info(
    f"99% quantile non-zero genes    : "
    f"{np.quantile(num_of_non_zero_genes, 0.99):.2f}"
)
logger.info(f"Max expression value           : {np.max(train_data)}")
logger.info(
    f"Avg non-zero expression value  : "
    f"{np.mean(train_data[np.nonzero(train_data)]):.4f}"
)
logger.info(
    f"99% quantile non-zero value    : "
    f"{np.quantile(train_data[np.nonzero(train_data)], 0.99):.4f}"
)
logger.info(f"Num of cell types              : {num_types}")

# 어휘 구성: 사전학습 유전자와 신규 유전자(Protein 토큰 등) 분리 처리
if config.load_model is None:
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
    gene_ids_pretrained = None
    pretrained_genes = []
else:
    pretrained_genes = [g for g in genes + special_tokens if g in old_vocab]
    new_genes = [g for g in genes + special_tokens if g not in old_vocab]
    gene_ids_pretrained = np.array(old_vocab(pretrained_genes), dtype=int)
    # 신규 토큰을 포함한 확장 어휘 구성
    # 참고: https://discuss.pytorch.org/t/expand-an-existing-embedding-and-linear-layer-nan-loss-value/55670/2
    vocab = Vocab(VocabPybind(pretrained_genes + new_genes, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    append_cls=True,
    include_zero_gene=config.include_zero_gene,
    mod_type=mod_type if config.use_mod else None,
    vocab_mod=vocab_mod if config.use_mod else None,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    append_cls=True,
    include_zero_gene=config.include_zero_gene,
    mod_type=mod_type if config.use_mod else None,
    vocab_mod=vocab_mod if config.use_mod else None,
)
logger.info(
    f"Train set: {tokenized_train['genes'].shape[0]} samples, "
    f"feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"Valid set: {tokenized_valid['genes'].shape[0]} samples, "
    f"feature length: {tokenized_valid['genes'].shape[1]}"
)


# ======================================================================
# 7. 모델 생성 및 사전학습 임베딩 이식 (Step 3: Load Pre-trained Model)
# 사전학습 모델은 RNA 토큰만 포함하므로, Protein 토큰을 위한
# 임베딩 레이어를 확장하고 RNA 임베딩 가중치만 이식
# ======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [수정] torch.load에 map_location 추가 (CPU 환경 안전 로드)
model_dict = torch.load(model_file, map_location=device)

ntokens = len(vocab)
model = MultiOmicTransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    do_mvc=config.GEPC,
    do_dab=config.DAR,
    use_batch_labels=config.use_batch_labels,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    n_input_bins=config.n_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=config.explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
    use_mod=config.use_mod,
    ntokens_mod=ntokens_mod if config.use_mod else None,
    vocab_mod=vocab_mod if config.use_mod else None,
)

# 사전학습 RNA 임베딩 가중치 이식 (신규 토큰 부분은 랜덤 초기화 유지)
if config.load_model is not None and gene_ids_pretrained is not None:
    with torch.no_grad():
        pretrained_emb = model_dict["encoder.embedding.weight"][gene_ids_pretrained, :]
        model.encoder.embedding.weight.data[: len(pretrained_genes), :] = pretrained_emb
        model.encoder.enc_norm.weight.data = model_dict["encoder.enc_norm.weight"]

model.to(device)
wandb.watch(model)
print(model)


# ======================================================================
# 8. 옵티마이저 및 손실함수 설정
# ======================================================================

# 활성화된 목표에 따라 손실함수 조건부 정의
criterion_gep_gepc = masked_mse_loss if (config.GEP and config.GEPC) else None
criterion_cls = nn.CrossEntropyLoss() if config.CLS else None
criterion_dab = nn.CrossEntropyLoss() if config.DAR else None

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.lr,
    eps=1e-4 if config.amp else 1e-8,
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1, gamma=config.schedule_ratio
)
# [수정] torch.cuda.amp → torch.amp (PyTorch 1.13+ 권장)
scaler = torch.amp.GradScaler("cuda", enabled=config.amp)


# ======================================================================
# 9. 파인튜닝 학습 루프 (Step 4: Fine-tune scGPT)
# ======================================================================

best_val_loss = float("inf")
best_model = None
best_model_epoch = 1  # val_loss가 갱신되지 않을 때를 대비한 초기값
define_wandb_metrcis()

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()

    train_data_pt, valid_data_pt = prepare_data(
        tokenized_train=tokenized_train,
        tokenized_valid=tokenized_valid,
        train_batch_labels=train_batch_labels,
        valid_batch_labels=valid_batch_labels,
        config=config,
        epoch=epoch,
        sort_seq_batch=config.per_seq_batch_sample,
    )
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=True,
        intra_domain_shuffle=False,
        drop_last=False,
        per_seq_batch_sample=config.per_seq_batch_sample,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
        per_seq_batch_sample=config.per_seq_batch_sample,
    )

    if config.do_train:
        train(
            model=model,
            loader=train_loader,
            vocab=vocab,
            criterion_gep_gepc=criterion_gep_gepc,
            criterion_dab=criterion_dab,
            criterion_cls=criterion_cls,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            logger=logger,
            epoch=epoch,
        )

    val_loss = evaluate(
        model=model,
        loader=valid_loader,
        vocab=vocab,
        criterion_gep_gepc=criterion_gep_gepc,
        criterion_dab=criterion_dab,
        criterion_cls=criterion_cls,
        device=device,
        config=config,
        epoch=epoch,
    )

    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss {val_loss:.4f} |"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model updated | loss: {best_val_loss:.4f}")

    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        logger.info(f"Saving model to {save_dir}")
        torch.save(
            best_model.state_dict(),
            save_dir / f"model_e{best_model_epoch}.pt",
        )

        results = eval_testdata(
            model=best_model,
            adata_t=adata_sorted if config.per_seq_batch_sample else adata,
            gene_ids=gene_ids,
            vocab=vocab,
            config=config,
            logger=logger,
            include_types=["cls"],
        )

        batch_umap_path = (
            save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png"
        )
        celltype_umap_path = (
            save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png"
        )
        results["batch_umap"].savefig(batch_umap_path, dpi=300)
        results["celltype_umap"].savefig(celltype_umap_path, dpi=300)

        # [수정] Figure 객체를 wandb.log에 직접 넘기지 않고 파일 경로로 전달
        metrics_to_log = {
            f"test/{k}": v
            for k, v in results.items()
            if k not in ("batch_umap", "celltype_umap")
        }
        metrics_to_log["test/batch_umap"] = wandb.Image(
            str(batch_umap_path),
            caption=f"batch avg_bio epoch {best_model_epoch}",
        )
        metrics_to_log["test/celltype_umap"] = wandb.Image(
            str(celltype_umap_path),
            caption=f"celltype avg_bio epoch {best_model_epoch}",
        )
        metrics_to_log["test/best_model_epoch"] = best_model_epoch
        wandb.log(metrics_to_log)
        wandb.log({"avg_bio": results.get("avg_bio", 0.0)})

    scheduler.step()


# ======================================================================
# 10. 최종 모델 저장 및 W&B 마무리 (Step 5)
# ======================================================================

torch.save(best_model.state_dict(), save_dir / "best_model.pt")
logger.info(f"Best model saved to {save_dir / 'best_model.pt'}")

artifact = wandb.Artifact("best_model", type="model")
artifact.add_file(str(save_dir / "best_model.pt"))
run.log_artifact(artifact)

run.finish()
wandb.finish()
gc.collect()
