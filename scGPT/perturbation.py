#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT: Fine-tuning Pre-trained Model for Perturbation Prediction

Environment Requirements:
  anndata==0.8.0      scanpy==1.9.1       torch==1.13.0
  numpy==1.21.6       pandas==1.3.5       scipy==1.7.3
  matplotlib==3.5.2   seaborn==0.11.2     scikit-learn==1.0.2
  umap-learn==0.5.3   scvi-tools==0.16.4  leidenalg==0.8.10
"""

import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경용 non-interactive 백엔드
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab

from gears import GEARS, PertData
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.loss import (
    criterion_neg_log_bernoulli,
    masked_mse_loss,
    masked_relative_error,
)
from scgpt.model import TransformerGenerator
from scgpt.tokenizer import pad_batch, tokenize_and_pad_batch, tokenize_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import compute_perturbation_metrics, map_raw_id_to_vocab_id, set_seed

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

set_seed(42)


# ======================================================================
# 1. 학습 설정 (Training Settings)
# ======================================================================

# 데이터 처리 설정
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 1536

# 학습 목표 플래그
MLM = True   # Masked Language Modeling (항상 활성화)
CLS = False  # Cell type classification
CCE = False  # Contrastive cell embedding
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity

# 사전학습 모델 경로
load_model = "./data/pretrain_human"
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# 옵티마이저 설정
lr = 1e-4
batch_size = 64
eval_batch_size = 64
epochs = 15
schedule_interval = 1
early_stop = 10
amp = True

# 모델 아키텍처 설정 (사전학습 모델 로드 시 args.json으로 덮어씌워짐)
embsize = 512
d_hid = 512
nlayers = 12
nhead = 8
n_layers_cls = 3
dropout = 0.0
use_fast_transformer = True

# 로깅 설정
log_interval = 100

# 데이터셋 및 평가 설정
data_name = "adamson"  # "norman" 또는 "adamson"
split = "simulation"

perts_to_plot = {
    "norman": ["SAMD1+ZBTB1"],
    "adamson": ["KCTD16+ctrl"],
}[data_name]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# 2. 저장 경로 및 로거 초기화
# ======================================================================

save_dir = Path(f"./results/perturbation/dev_perturb_{data_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving to {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")


# ======================================================================
# 3. 데이터 로드 및 전처리
# ======================================================================

pert_data = PertData("./data")
pert_data.load(data_name=data_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)


# ======================================================================
# 4. 어휘(Vocab) 및 유전자 ID 설정
# ======================================================================

if load_model is not None:
    model_dir = Path(load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1
        for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(
        f"Match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    genes = pert_data.adata.var["gene_name"].tolist()

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, "
        f"model args will override config {model_config_file}."
    )
    # 사전학습 모델 설정으로 덮어씌우기
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

else:
    genes = pert_data.adata.var["gene_name"].tolist()
    vocab = Vocab(VocabPybind(genes + special_tokens, None))

vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(
    [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes],
    dtype=int,
)
n_genes = len(genes)


# ======================================================================
# 5. 모델 생성 및 사전학습 가중치 로드
# ======================================================================

ntokens = len(vocab)
model = TransformerGenerator(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    pert_pad_id=pert_pad_id,
    use_fast_transformer=use_fast_transformer,
)

if load_model is not None:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file, map_location=device)

    if load_param_prefixs is not None:
        # 지정된 prefix로 시작하고 model_dict에 존재하며 shape이 일치하는 파라미터만 로드
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if any(k.startswith(prefix) for prefix in load_param_prefixs)
            and k in model_dict
            and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading param: {k} | shape: {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        try:
            model.load_state_dict(pretrained_dict)
            logger.info(f"Loaded all model params from {model_file}")
        except RuntimeError:
            # shape이 맞는 파라미터만 부분 로드
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading param: {k} | shape: {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

model.to(device)


# ======================================================================
# 6. 학습 함수 정의
# ======================================================================

criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)

scaler = torch.cuda.amp.GradScaler(enabled=amp)


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """한 에포크 동안 모델을 학습합니다."""
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()
    num_batches = len(train_loader)

    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)

        x: torch.Tensor = batch_data.x             # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y           # (batch_size, n_genes)

        # 입력 유전자 ID 선택
        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(
                    len(input_gene_ids), device=device
                )[:max_seq_len]

            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]
            masked_positions = torch.ones_like(input_values, dtype=torch.bool)
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if w:
                logger.warning(
                    f"Infinite gradient detected. Current scaler scale: "
                    f"{scaler.get_scale()}. Safe to ignore if it resolves automatically."
                )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mse.item()

        if batch % log_interval == 0 and batch > 0:
            cur_lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {cur_lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {total_loss / log_interval:5.2f} | "
                f"mse {total_mse / log_interval:5.2f} |"
            )
            total_loss, total_mse = 0.0, 0.0
            start_time = time.time()


def eval_perturb(
    loader: DataLoader,
    model: TransformerGenerator,
    device: torch.device,
) -> Dict:
    """주어진 데이터 로더로 모델을 추론 모드에서 실행합니다."""
    model.eval()
    model.to(device)

    pert_cat, pred, truth, pred_de, truth_de = [], [], [], [], []

    for batch in loader:
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            for i, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[i, de_idx])
                truth_de.append(t[i, de_idx])

    pred = torch.stack(pred)
    truth = torch.stack(truth)
    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)

    # [수정] np.float deprecated → np.float32
    return {
        "pert_cat": np.array(pert_cat),
        "pred": pred.detach().cpu().numpy().astype(np.float32),
        "truth": truth.detach().cpu().numpy().astype(np.float32),
        "pred_de": pred_de.detach().cpu().numpy().astype(np.float32),
        "truth_de": truth_de.detach().cpu().numpy().astype(np.float32),
    }


# ======================================================================
# 7. 학습 루프 (Training Loop)
# ======================================================================

best_val_corr = 0.0
best_model = None
patience = 0

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()

    train(model, pert_data.dataloader["train_loader"])

    val_res = eval_perturb(pert_data.dataloader["val_loader"], model, device)
    val_metrics = compute_perturbation_metrics(
        val_res,
        pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"],
    )
    elapsed = time.time() - epoch_start_time
    logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s |")
    logger.info(f"val_metrics at epoch {epoch}: {val_metrics}")

    val_score = val_metrics["pearson"]
    if val_score > best_val_corr:
        best_val_corr = val_score
        best_model = copy.deepcopy(model)
        logger.info(f"Best model updated | pearson: {val_score:.4f}")
        patience = 0
    else:
        patience += 1
        if patience >= early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    scheduler.step()

torch.save(best_model.state_dict(), save_dir / "best_model.pt")
logger.info(f"Best model saved to {save_dir / 'best_model.pt'}")


# ======================================================================
# 8. 추론 및 시각화 함수 정의
# ======================================================================

def predict(
    model: TransformerGenerator,
    pert_list: List[List[str]],
    pool_size: Optional[int] = None,
) -> Dict:
    """
    주어진 perturbation 목록에 대한 유전자 발현값을 예측합니다.

    Args:
        model: 예측에 사용할 TransformerGenerator 모델.
        pert_list: 예측할 perturbation 목록. 각 항목은 유전자 이름의 리스트.
        pool_size: 각 perturbation당 사용할 control cell 수.
                   None이면 모든 control cell 사용.

    Returns:
        perturbation 이름을 키로, 예측 발현값 배열을 값으로 하는 딕셔너리.
    """
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    pool_size = pool_size or len(ctrl_adata.obs)

    gene_list = pert_data.gene_names.values.tolist()
    for pert in pert_list:
        for gene in pert:
            if gene not in gene_list:
                raise ValueError(
                    f"Gene '{gene}' is not in the perturbation graph. "
                    "Please select from GEARS.gene_list."
                )

    model.eval()
    device = next(model.parameters()).device
    results_pred = {}

    with torch.no_grad():
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(
                preds.detach().cpu().numpy(), axis=0
            )

    return results_pred


def plot_perturbation(
    model: nn.Module,
    query: str,
    save_file: Optional[str] = None,
    pool_size: Optional[int] = None,
) -> matplotlib.figure.Figure:
    """
    perturbation 결과와 실제값을 비교하는 박스플롯을 생성합니다.

    Args:
        model: 예측에 사용할 모델.
        query: 시각화할 perturbation 조건명 (예: "KCTD16+ctrl").
        save_file: 그림을 저장할 파일 경로. None이면 저장하지 않음.
        pool_size: 예측에 사용할 control cell 수.

    Returns:
        생성된 matplotlib Figure 객체.
    """
    import seaborn as sns
    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    adata = pert_data.adata
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    genes = [
        gene_raw2id[i]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]

    truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]

    parts = query.split("+")
    if parts[1] == "ctrl":
        pred = predict(model, [[parts[0]]], pool_size=pool_size)
        pred = pred[parts[0]][de_idx]
    else:
        pred = predict(model, [parts], pool_size=pool_size)
        pred = pred["_".join(parts)][de_idx]

    ctrl_means = (
        adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values
    )
    pred = pred - ctrl_means
    truth = truth - ctrl_means

    fig, ax = plt.subplots(figsize=(16.5, 4.5))
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))
    for i in range(pred.shape[0]):
        plt.scatter(i + 1, pred[i], color="red")
    plt.axhline(0, linestyle="dashed", color="green")
    ax.xaxis.set_ticklabels(genes, rotation=90)
    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine()

    if save_file:
        fig.savefig(save_file, bbox_inches="tight", transparent=False)

    return fig


# ======================================================================
# 9. 평가 (Evaluation)
# ======================================================================

# perturbation 시각화
for p in perts_to_plot:
    plot_perturbation(
        best_model, p, pool_size=300, save_file=str(save_dir / f"{p}.png")
    )

# 테스트셋 평가
test_res = eval_perturb(pert_data.dataloader["test_loader"], best_model, device)
test_metrics = compute_perturbation_metrics(
    test_res,
    pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"],
)
print(test_metrics)

with open(save_dir / "test_metrics.json", "w") as f:
    json.dump(test_metrics, f)

# Subgroup 분석
deeper_res = deeper_analysis(pert_data.adata, test_res)
non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

metrics = ["pearson_delta", "pearson_delta_de"]
metrics_non_dropout = [
    "pearson_delta_top20_de_non_dropout",
    "pearson_top20_de_non_dropout",
]

subgroup_analysis = {
    name: {m: [] for m in metrics + metrics_non_dropout}
    for name in pert_data.subgroup["test_subgroup"].keys()
}

for name, pert_list in pert_data.subgroup["test_subgroup"].items():
    for pert in pert_list:
        for m in metrics:
            subgroup_analysis[name][m].append(deeper_res[pert][m])
        for m in metrics_non_dropout:
            subgroup_analysis[name][m].append(non_dropout_res[pert][m])

for name, result in subgroup_analysis.items():
    for m, values in result.items():
        logger.info(f"test_{name}_{m}: {np.mean(values):.4f}")
