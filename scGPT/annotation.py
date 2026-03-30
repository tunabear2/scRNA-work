#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT Fine-tuning for Cell-type Annotation
===========================================
Multiple Sclerosis 데이터셋을 사용하여 사전학습된 scGPT 모델을
세포 유형 분류(cell-type annotation) 태스크로 파인튜닝합니다.

파이프라인 요약:
    1. 하이퍼파라미터 설정
    2. 데이터 로드 및 전처리
    3. 사전학습 모델 로드
    4. 태스크별 목적함수로 파인튜닝
    5. 평가 및 추론

디렉토리 구조 (실행 위치: ~/DW/scGPT/):
    ./data/annotation/       - c_data.h5ad, filtered_ms_adata.h5ad
    ./data/pretrain/         - args.json, best_model.pt, vocab.json
    ./scgpt/                 - scGPT 패키지
"""

import copy
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import types
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from anndata import AnnData
import scanpy as sc
import scvi
import seaborn as sns
from scipy.sparse import issparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

# scGPT 패키지 경로 (annotation.py가 프로젝트 루트에 위치)
sys.path.insert(0, ".")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

# ── wandb shim ──────────────────────────────────────────────────────────
# wandb가 설치되어 있지 않거나 로그인되지 않은 경우 더미 객체로 대체
class _WandbShim:
    """로컬 실행용 wandb 최소 대체 객체."""

    class Settings:
        def __init__(self, **kwargs):
            pass

    class _Run:
        config = None
        def log(self, d, **kw): pass
        def log_artifact(self, a): pass
        def finish(self): pass

    config = None

    @staticmethod
    def init(config=None, project=None, reinit=False, settings=None):
        run = _WandbShim._Run()
        run.config = types.SimpleNamespace(**config) if isinstance(config, dict) else config
        _WandbShim.config = run.config
        return run

    @staticmethod
    def finish(): pass

    @staticmethod
    def log(d, **kw): pass

    @staticmethod
    def watch(*args, **kwargs): pass

    @staticmethod
    def define_metric(*args, **kwargs): pass

    @staticmethod
    def Image(path, **kwargs):
        return path

try:
    import wandb as _real_wandb
    _real_wandb.login()
    wandb = _real_wandb
except Exception:
    wandb = _WandbShim()

# ── 글로벌 설정 ────────────────────────────────────────────────────────
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")


# ========================================================================
# 1. 하이퍼파라미터 설정
# ========================================================================
hyperparameter_defaults = dict(
    seed=0,
    dataset_name="ms",
    do_train=True,
    load_model="./data/pretrain_human",
    mask_ratio=0.0,
    epochs=20,
    n_bins=51,
    MVC=False,
    ecs_thres=0.0,
    dab_weight=0.0,
    lr=1e-4,
    batch_size=32,
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.2,
    schedule_ratio=0.9,
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=False,
    DSBN=False,
)

run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
print(config)
set_seed(config.seed)

# ── 입력/전처리 설정 ───────────────────────────────────────────────────
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"
include_zero_gene = config.include_zero_gene
max_seq_len = 3001
n_bins = config.n_bins

input_style = "binned"
output_style = "binned"

# ── 학습 목적함수 설정 ─────────────────────────────────────────────────
MLM = False
CLS = True
ADV = False
CCE = False
MVC = config.MVC
ECS = config.ecs_thres > 0
DAB = False
INPUT_BATCH_LABELS = False
input_emb_style = "continuous"
cell_emb_style = "cls"
adv_E_delay_epochs = 0
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight

explicit_zero_prob = MLM and include_zero_gene
do_sample_in_train = False and explicit_zero_prob
per_seq_batch_sample = False

# ── 옵티마이저 설정 ───────────────────────────────────────────────────
lr = config.lr
lr_ADV = 1e-3
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1

# ── 모델 아키텍처 설정 ────────────────────────────────────────────────
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"
embsize = config.layer_size
d_hid = config.layer_size
nlayers = config.nlayers
nhead = config.nhead
dropout = config.dropout

# ── 로깅 설정 ─────────────────────────────────────────────────────────
log_interval = 100
save_eval_interval = config.save_eval_interval
do_eval_scib_metrics = True

# ── 입력 표현 검증 ────────────────────────────────────────────────────
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]

if input_style == "binned" and input_emb_style == "scaling":
    raise ValueError("input_emb_style 'scaling'은 binned 입력에서 지원되지 않습니다.")
if input_style in ("log1p", "normed_raw") and input_emb_style == "category":
    raise ValueError("input_emb_style 'category'는 log1p/normed_raw 입력에서 지원되지 않습니다.")

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV와 DAB는 동시에 True일 수 없습니다.")
DAB_separate_optim = True if DAB > 1 else False

# ── 저장 디렉토리 ────────────────────────────────────────────────────
dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


# ========================================================================
# 2. 데이터 로드 및 전처리
# ========================================================================
if dataset_name == "ms":
    data_dir = Path("./data/annotation")
    adata = sc.read(data_dir / "c_data.h5ad")
    adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")

    adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
    adata.var.set_index(adata.var["gene_name"], inplace=True)
    adata_test.var.set_index(adata_test.var["gene_name"], inplace=True)

    data_is_raw = False
    filter_gene_by_counts = False
    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")

# 배치 및 세포유형 레이블 생성
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels

celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
celltypes = adata.obs["celltype"].unique()
num_types = len(np.unique(celltype_id_labels))
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
adata.obs["celltype_id"] = celltype_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

# ── 사전학습 모델 vocab 로드 ──────────────────────────────────────────
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # vocab에 있는 유전자만 필터링
    adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # 모델 설정 로드
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(f"Resume model from {model_file}, the model args will override the config {model_config_file}.")
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

# ── 전처리 ────────────────────────────────────────────────────────────
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=filter_gene_by_counts,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=data_is_raw,
    result_log1p_key="X_log1p",
    subset_hvg=False,
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,
    result_binned_key="X_binned",
)

adata_test = adata[adata.obs["str_batch"] == "1"]
adata = adata[adata.obs["str_batch"] == "0"]

preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)

# ── 학습/검증 데이터 분할 ──────────────────────────────────────────────
input_layer_key = {"normed_raw": "X_normed", "log1p": "X_normed", "binned": "X_binned"}[input_style]

all_counts = (
    adata.layers[input_layer_key].toarray()
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()
celltypes_labels = np.array(adata.obs["celltype_id"].tolist())
batch_ids = np.array(adata.obs["batch_id"].tolist())
num_batch_types = len(set(batch_ids))

(
    train_data, valid_data,
    train_celltype_labels, valid_celltype_labels,
    train_batch_labels, valid_batch_labels,
) = train_test_split(all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True)

# ── Vocab / Gene IDs ──────────────────────────────────────────────────
if config.load_model is None:
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

# ── 토크나이즈 ────────────────────────────────────────────────────────
tokenized_train = tokenize_and_pad_batch(
    train_data, gene_ids, max_len=max_seq_len, vocab=vocab,
    pad_token=pad_token, pad_value=pad_value,
    append_cls=True, include_zero_gene=include_zero_gene,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data, gene_ids, max_len=max_seq_len, vocab=vocab,
    pad_token=pad_token, pad_value=pad_value,
    append_cls=True, include_zero_gene=include_zero_gene,
)
logger.info(f"train set: {tokenized_train['genes'].shape[0]} samples, feature length: {tokenized_train['genes'].shape[1]}")
logger.info(f"valid set: {tokenized_valid['genes'].shape[0]} samples, feature length: {tokenized_valid['genes'].shape[1]}")


# ========================================================================
# 헬퍼 함수: 데이터 준비 및 DataLoader
# ========================================================================
def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"], mask_ratio=mask_ratio,
        mask_value=mask_value, pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"], mask_ratio=mask_ratio,
        mask_value=mask_value, pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: "
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}"
    )

    input_gene_ids_train, input_gene_ids_valid = tokenized_train["genes"], tokenized_valid["genes"]
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = tokenized_train["values"], tokenized_valid["values"]

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()
    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train, "values": input_values_train,
        "target_values": target_values_train, "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid, "values": input_values_valid,
        "target_values": target_values_valid, "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }
    return train_data_pt, valid_data_pt


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        return DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets, batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers, pin_memory=True,
        )

    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers, pin_memory=True,
    )


# ========================================================================
# 3. 사전학습 모델 로드
# ========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)

model = TransformerModel(
    ntokens, embsize, nhead, d_hid, nlayers,
    nlayers_cls=3,
    n_cls=num_types if CLS else 1,
    vocab=vocab, dropout=dropout,
    pad_token=pad_token, pad_value=pad_value,
    do_mvc=MVC, do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)

if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
        logger.info(f"Loading all model params from {model_file}")
    except Exception:
        # 크기가 맞는 파라미터만 로드
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

pre_freeze_param_count = sum(
    dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()
)

# 선택적 가중치 동결
for name, para in model.named_parameters():
    if config.freeze and "encoder" in name and "transformer_encoder" not in name:
        logger.info(f"Freezing: {name}")
        para.requires_grad = False

post_freeze_param_count = sum(
    dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()
)
logger.info(f"Pre-freeze params: {pre_freeze_param_count:,}")
logger.info(f"Post-freeze params: {post_freeze_param_count:,}")
wandb.log({
    "info/pre_freeze_param_count": pre_freeze_param_count,
    "info/post_freeze_param_count": post_freeze_param_count,
})

model.to(device)
wandb.watch(model)

if ADV:
    discriminator = AdversarialDiscriminator(d_model=embsize, n_cls=num_batch_types).to(device)


# ========================================================================
# 옵티마이저 / 스케줄러 / 스케일러
# ========================================================================
criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=config.schedule_ratio)

if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(optimizer_dab, schedule_interval, gamma=config.schedule_ratio)
if ADV:
    criterion_adv = nn.CrossEntropyLoss()
    optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
    scheduler_E = torch.optim.lr_scheduler.StepLR(optimizer_E, schedule_interval, gamma=config.schedule_ratio)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, schedule_interval, gamma=config.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


# ========================================================================
# 학습 / 평가 함수
# ========================================================================
def train(model: nn.Module, loader: DataLoader) -> None:
    """한 에포크 학습."""
    model.train()
    total_loss = total_mse = total_cls = total_cce = 0.0
    total_mvc = total_ecs = total_dab = 0.0
    total_adv_E = total_adv_D = 0.0
    total_zero_log_prob = total_mvc_zero_log_prob = 0.0
    total_error = 0.0
    start_time = time.time()
    num_batches = len(loader)

    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids, input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS,
                do_sample=do_sample_in_train,
            )

            masked_positions = input_values.eq(mask_value)
            loss = 0.0
            metrics_to_log = {}

            if MLM:
                loss_mse = criterion(output_dict["mlm_output"], target_values, masked_positions)
                loss = loss + loss_mse
                metrics_to_log["train/mse"] = loss_mse.item()
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(output_dict["mlm_zero_probs"], target_values, masked_positions)
                loss = loss + loss_zero_log_prob
                metrics_to_log["train/nzlp"] = loss_zero_log_prob.item()
            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log["train/cls"] = loss_cls.item()
                error_rate = 1 - (output_dict["cls_output"].argmax(1) == celltype_labels).sum().item() / celltype_labels.size(0)
            if CCE:
                loss_cce = 10 * output_dict["loss_cce"]
                loss = loss + loss_cce
                metrics_to_log["train/cce"] = loss_cce.item()
            if MVC:
                loss_mvc = criterion(output_dict["mvc_output"], target_values, masked_positions)
                loss = loss + loss_mvc
                metrics_to_log["train/mvc"] = loss_mvc.item()
            if MVC and explicit_zero_prob:
                loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(output_dict["mvc_zero_probs"], target_values, masked_positions)
                loss = loss + loss_mvc_zero_log_prob
                metrics_to_log["train/mvc_nzlp"] = loss_mvc_zero_log_prob.item()
            if ECS:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log["train/ecs"] = loss_ecs.item()
            if DAB:
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + dab_weight * loss_dab
                metrics_to_log["train/dab"] = loss_dab.item()

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. Current scale: {scaler.get_scale()}. "
                    "This warning can be ignored if it stops after autoscaling."
                )
        scaler.step(optimizer)
        scaler.update()

        if ADV:
            output_dict = model(
                input_gene_ids, input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS,
                do_sample=do_sample_in_train,
            )
            loss_adv_D = criterion_adv(discriminator(output_dict["cell_emb"].detach()), batch_labels)
            if epoch > adv_D_delay_epochs:
                discriminator.zero_grad()
                loss_adv_D.backward()
                optimizer_D.step()
            loss_adv_E = -criterion_adv(discriminator(output_dict["cell_emb"]), batch_labels)
            if epoch > adv_E_delay_epochs:
                model.zero_grad()
                discriminator.zero_grad()
                loss_adv_E.backward()
                optimizer_E.step()

        wandb.log(metrics_to_log)

        total_loss += loss.item()
        total_mse += loss_mse.item() if MLM else 0.0
        total_cls += loss_cls.item() if CLS else 0.0
        total_cce += loss_cce.item() if CCE else 0.0
        total_mvc += loss_mvc.item() if MVC else 0.0
        total_ecs += loss_ecs.item() if ECS else 0.0
        total_dab += loss_dab.item() if DAB else 0.0
        total_adv_E += loss_adv_E.item() if ADV else 0.0
        total_adv_D += loss_adv_D.item() if ADV else 0.0
        total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
        total_mvc_zero_log_prob += loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0
        total_error += error_rate

        if batch % log_interval == 0 and batch > 0:
            cur_lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_error = total_error / log_interval
            log_msg = (
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {cur_lr:05.4f} | ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.2f} | "
            )
            if CLS:
                log_msg += f"cls {total_cls / log_interval:5.2f} | err {cur_error:5.2f} | "
            if MLM:
                log_msg += f"mse {total_mse / log_interval:5.2f} | "
            logger.info(log_msg)

            total_loss = total_mse = total_cls = total_cce = 0.0
            total_mvc = total_ecs = total_dab = 0.0
            total_adv_E = total_adv_D = 0.0
            total_zero_log_prob = total_mvc_zero_log_prob = 0.0
            total_error = 0.0
            start_time = time.time()


def define_wandb_metrics():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/err", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """검증 데이터로 모델 평가."""
    model.eval()
    total_loss = total_error = total_dab = 0.0
    total_num = 0
    predictions = []
    labels_all = []

    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids, input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS, CCE=False, MVC=False, ECS=False,
                    do_sample=do_sample_in_train,
                )
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)
                if DAB:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            pred = output_values.argmax(1)
            n = len(input_gene_ids)
            total_loss += loss.item() * n
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / n) * n
            total_dab += loss_dab.item() * n if DAB else 0.0
            total_num += n
            predictions.append(pred.cpu().numpy())
            labels_all.append(celltype_labels.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    val_precision = precision_score(labels_all, predictions, average="macro", zero_division=0)
    val_recall = recall_score(labels_all, predictions, average="macro", zero_division=0)
    val_f1 = f1_score(labels_all, predictions, average="macro", zero_division=0)
    val_acc = accuracy_score(labels_all, predictions)

    wandb.log({
        "valid/accuracy": val_acc,
        "valid/precision": val_precision,
        "valid/recall": val_recall,
        "valid/f1": val_f1,
        "valid/mse": total_loss / total_num,
        "valid/err": total_error / total_num,
        "valid/dab": total_dab / total_num,
        "valid/sum_mse_dab": (total_loss + dab_weight * total_dab) / total_num,
        "epoch": epoch,
    })

    if return_raw:
        return predictions
    return total_loss / total_num, total_error / total_num, val_acc, val_precision, val_recall, val_f1


# ========================================================================
# 4. 파인튜닝
# ========================================================================
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrics()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)

    train_loader = prepare_dataloader(
        train_data_pt, batch_size=batch_size,
        shuffle=False, intra_domain_shuffle=True, drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt, batch_size=eval_batch_size,
        shuffle=False, intra_domain_shuffle=False, drop_last=False,
    )

    if config.do_train:
        train(model, loader=train_loader)

    val_loss, val_err, val_acc, val_precision, val_recall, val_f1 = evaluate(model, loader=valid_loader)
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.4f} | err {val_err:5.4f} | acc {val_acc:5.4f} | precision {val_precision:5.4f} | recall {val_recall:5.4f} | f1 {val_f1:5.4f}")
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    scheduler.step()
    if DAB_separate_optim:
        scheduler_dab.step()
    if ADV:
        scheduler_D.step()
        scheduler_E.step()


# ========================================================================
# 5. 테스트 추론 및 결과 저장
# ========================================================================
def test(model: nn.Module, adata: AnnData) -> Tuple[np.ndarray, np.ndarray, dict]:
    """테스트 데이터로 추론 후 분류 지표 계산."""
    all_counts = (
        adata.layers[input_layer_key].toarray()
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    celltypes_labels = np.array(adata.obs["celltype_id"].tolist())
    batch_ids = np.array(adata.obs["batch_id"].tolist())

    tokenized_test = tokenize_and_pad_batch(
        all_counts, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=pad_value,
        append_cls=True, include_zero_gene=include_zero_gene,
    )
    input_values_test = random_mask_value(
        tokenized_test["values"], mask_ratio=mask_ratio,
        mask_value=mask_value, pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }
    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size, shuffle=False, drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
        pin_memory=True,
    )

    model.eval()
    predictions = evaluate(model, loader=test_loader, return_raw=True)

    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    logger.info(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Macro F1: {macro_f1:.3f}")

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }
    return predictions, celltypes_labels, results


# ── 테스트 실행 ────────────────────────────────────────────────────────
predictions, labels, results = test(best_model, adata_test)
adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]

# UMAP 시각화
palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3
palette_ = {c: palette_[i] for i, c in enumerate(celltypes)}

with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": 300}):
    sc.pl.umap(adata_test_raw, color=["celltype", "predictions"], palette=palette_, show=False)
    plt.savefig(save_dir / "results.png", dpi=300)

# 결과 저장
save_dict = {"predictions": predictions, "labels": labels, "results": results, "id_maps": id2type}
with open(save_dir / "results.pkl", "wb") as f:
    pickle.dump(save_dict, f)

results["test/cell_umap"] = wandb.Image(str(save_dir / "results.png"), caption=f"macro f1 {results['test/macro_f1']:.3f}")
wandb.log(results)

# ── Confusion Matrix ──────────────────────────────────────────────────
celltypes_list = list(celltypes)
for ct in set(id2type[p] for p in predictions):
    if ct not in celltypes_list:
        celltypes_list.remove(ct)

cm = confusion_matrix(labels, predictions)
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
cm = pd.DataFrame(cm, index=celltypes_list[:cm.shape[0]], columns=celltypes_list[:cm.shape[1]])

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
plt.savefig(save_dir / "confusion_matrix.png", dpi=300)

results["test/confusion_matrix"] = wandb.Image(str(save_dir / "confusion_matrix.png"), caption="confusion matrix")
wandb.log(results)

# ── 최종 모델 저장 ────────────────────────────────────────────────────
torch.save(best_model.state_dict(), save_dir / "model.pt")
logger.info(f"Best model (epoch {best_model_epoch}) saved to {save_dir / 'model.pt'}")
wandb.finish()
