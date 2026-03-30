"""
scGPT Fine-tuning for Multiomic Integration (BMMC CITE-seq)

RNA + Protein(ADT) 데이터를 통합하여 사전 학습된 scGPT 모델을 파인튜닝합니다.

Steps:
    1. 하이퍼파라미터 설정
    2. 데이터 로드 및 전처리
    3. 사전 학습 모델 로드
    4. 파인튜닝
    5. 평가

Requirements:
    - BMMC_processed.h5ad    : CITE-seq BMMC 데이터셋
    - ./data/pretrain_human/ : 사전 학습된 scGPT whole-body 모델
"""

import copy
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from anndata import AnnData
from scipy.sparse import issparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
import scanpy as sc

sys.path.insert(0, "../")
import scgpt as scg
from scgpt import prepare_data, prepare_dataloader, define_wandb_metrcis, evaluate, eval_testdata, train
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import MultiOmicTransformerModel
from scgpt.loss import masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# W&B 설정: 로그인 성공 시 온라인, 실패 시 오프라인 모드로 자동 전환
# ──────────────────────────────────────────────
import wandb

os.environ["WANDB_MODE"] = "offline"  # 기본값: 오프라인
USE_WANDB = True
try:
    result = wandb.login(timeout=10)
    if result:
        del os.environ["WANDB_MODE"]
        print("W&B 온라인 모드로 실행합니다.")
    else:
        print("W&B 로그인 실패 — 오프라인 모드로 실행합니다. 나중에 'wandb sync'로 업로드 가능합니다.")
except Exception:
    print("W&B 로그인 실패 — 오프라인 모드로 실행합니다. 나중에 'wandb sync'로 업로드 가능합니다.")


# ──────────────────────────────────────────────
# Step 1. 하이퍼파라미터 설정
# ──────────────────────────────────────────────
HYPERPARAMS = dict(
    task                = "multiomic",
    seed                = 42,
    dataset_name        = "BMMC",
    do_train            = True,
    load_model          = "./data/pretrain_human",  # 사전 학습 모델 경로
    freeze              = False,
    GEP                 = True,    # Gene Expression Prediction
    GEPC                = True,    # Gene Expression Prediction (cell-level)
    CLS                 = False,
    ESC                 = False,
    DAR                 = True,    # Domain Adaptive Regularization (배치 보정)
    DSBN                = False,   # Domain-specific Batch Normalization
    mask_ratio          = 0.4,
    explicit_zero_prob  = False,
    ecs_thres           = 0,
    dab_weight          = 1.0,
    use_batch_labels    = True,
    use_mod             = True,    # 모달리티 인식 학습
    per_seq_batch_sample= False,
    epochs              = 25,
    input_layer_key     = "X_binned",
    n_bins              = 51,
    n_hvg               = 1200,
    n_hvp               = 4000,
    max_seq_len         = 4001,    # n_hvg + 1
    lr                  = 1e-3,
    batch_size          = 16,
    layer_size          = 512,
    nlayers             = 4,
    nhead               = 8,
    dropout             = 0.2,
    schedule_ratio      = 0.95,
    save_eval_interval  = 5,
    log_interval        = 100,
    fast_transformer    = False,
    pre_norm            = False,
    amp                 = True,
    pad_token           = "<pad>",
    mask_value          = -1,
    pad_value           = -2,
    include_zero_gene   = False,
)


def get_config():
    """W&B 사용 여부에 따라 config 객체 반환."""
    if USE_WANDB:
        run = wandb.init(
            config   = HYPERPARAMS,
            project  = "scGPT",
            reinit   = True,
            settings = wandb.Settings(start_method="fork"),
        )
        return wandb.config, run
    else:
        return SimpleNamespace(**HYPERPARAMS), None


# ──────────────────────────────────────────────
# Step 2. 데이터 로드 및 전처리
# ──────────────────────────────────────────────
def load_bmmc(config):
    """BMMC CITE-seq 데이터 로드 및 RNA/Protein 분리."""
    adata = sc.read("./data/BMMC_processed.h5ad")

    # 3명 donor, 17종 세포 유형만 사용
    adata = adata[
        adata.obs.DonorID.isin([10886, 11466, 12710]) &
        adata.obs.cell_type.isin(np.unique(adata.obs.cell_type.values)[:17])
    ]

    adata.obs["celltype"]  = adata.obs["cell_type"].astype(str).astype("category")
    adata.var["gene_name"] = adata.var.index.tolist()

    le = preprocessing.LabelEncoder()
    adata.obs["batch_id"]  = le.fit_transform(adata.obs["batch"].values)
    adata.obs["str_batch"] = adata.obs["batch_id"].astype("category")

    # Protein(ADT) 분리
    adata_protein = adata[:, adata.var.feature_types.isin(["ADT"])].copy()
    adata_protein.var.index = ["p_" + i for i in adata_protein.var.index]

    # RNA 분리
    adata = adata[:, adata.var.feature_types.isin(["GEX"])].copy()

    data_is_raw = False
    return adata, adata_protein, data_is_raw


def build_modality_info(adata, adata_protein, special_tokens):
    """모달리티(RNA/Protein) 정보 생성."""
    gene_rna_df     = pd.DataFrame(index=adata.var.index.tolist())
    gene_protein_df = pd.DataFrame(index=adata_protein.var.index.tolist())
    gene_rna_df["mod"]     = "RNA"
    gene_protein_df["mod"] = "Protein"

    gene_loc_df = pd.concat([gene_rna_df, gene_protein_df])
    gene_loc_df["mod"] = gene_loc_df["mod"].astype("category")

    vocab_mod = Vocab(VocabPybind(
        np.unique(gene_loc_df["mod"]).tolist() + special_tokens, None
    ))
    vocab_mod.set_default_index(vocab_mod["<pad>"])

    return gene_loc_df, vocab_mod


def preprocess_rna(adata, config, data_is_raw):
    preprocessor = Preprocessor(
        use_key               = "X",
        filter_gene_by_counts = 1,
        filter_cell_by_counts = 1,
        normalize_total       = 1e4,
        result_normed_key     = "X_normed",
        log1p                 = data_is_raw,
        result_log1p_key      = "X_log1p",
        subset_hvg            = config.n_hvg,
        hvg_flavor            = "seurat_v3" if data_is_raw else "cell_ranger",
        binning               = config.n_bins,
        result_binned_key     = "X_binned",
    )
    preprocessor(adata, batch_key=None)
    return adata


def preprocess_protein(adata_protein, config):
    preprocessor = Preprocessor(
        use_key               = "X",
        filter_gene_by_counts = 0,
        filter_cell_by_counts = False,
        normalize_total       = False,
        result_normed_key     = "X_normed",
        log1p                 = False,
        result_log1p_key      = "X_log1p",
        subset_hvg            = False,
        hvg_flavor            = None,
        binning               = config.n_bins,
        result_binned_key     = "X_binned",
    )
    preprocessor(adata_protein, batch_key=None)
    return adata_protein


def combine_modalities(adata, adata_protein):
    """RNA + Protein 데이터를 하나의 AnnData로 합침."""
    data_combined = np.concatenate(
        [adata.layers["X_binned"], adata_protein.layers["X_binned"]], axis=1
    )
    combined = AnnData(
        X      = data_combined,
        obs    = adata.obs,
        var    = pd.DataFrame(index=adata.var_names.tolist() + adata_protein.var_names.tolist()),
        layers = {"X_binned": data_combined},
    )
    combined.var["gene_name"] = combined.var.index.tolist()
    return combined


# ──────────────────────────────────────────────
# Step 3. 사전 학습 모델 및 어휘 로드
# ──────────────────────────────────────────────
def load_vocab(config, adata, special_tokens, logger):
    model_dir         = Path(config.load_model)
    model_file        = model_dir / "best_model.pt"
    model_config_file = model_dir / "args.json"
    vocab_file        = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for token in special_tokens:
        if token not in vocab:
            vocab.append_token(token)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    n_matched = np.sum(adata.var["id_in_vocab"] >= 0)
    logger.info(f"vocab 매칭 유전자: {n_matched}/{len(adata.var)} (vocab 크기: {len(vocab)})")

    return vocab, model_file


def build_gene_vocab(genes, special_tokens, old_vocab):
    """사전 학습 vocab을 기반으로 새 vocab 생성 및 유전자 ID 반환."""
    pretrained_genes    = [g for g in genes + special_tokens if g in old_vocab]
    new_genes           = [g for g in genes + special_tokens if g not in old_vocab]
    gene_ids_pretrained = np.array(old_vocab(pretrained_genes), dtype=int)

    vocab = Vocab(VocabPybind(pretrained_genes + new_genes, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    return vocab, gene_ids, pretrained_genes, gene_ids_pretrained


# ──────────────────────────────────────────────
# Step 4. 모델 구성
# ──────────────────────────────────────────────
def build_model(config, vocab, model_file, device,
                num_batch_types, ntokens_mod, vocab_mod,
                pretrained_genes, gene_ids_pretrained):
    model_dict = torch.load(model_file, map_location=device)

    with open(Path(config.load_model) / "args.json") as f:
        model_cfg = json.load(f)

    embsize = model_cfg["embsize"]
    nhead   = model_cfg["nheads"]
    d_hid   = model_cfg["d_hid"]
    nlayers = model_cfg["nlayers"]

    model = MultiOmicTransformerModel(
        ntoken               = len(vocab),
        d_model              = embsize,
        nhead                = nhead,
        d_hid                = d_hid,
        nlayers              = nlayers,
        vocab                = vocab,
        dropout              = config.dropout,
        pad_token            = config.pad_token,
        pad_value            = config.pad_value,
        do_mvc               = config.GEPC,
        do_dab               = config.DAR,
        use_batch_labels     = config.use_batch_labels,
        num_batch_labels     = num_batch_types,
        domain_spec_batchnorm= config.DSBN,
        n_input_bins         = config.n_bins,
        ecs_threshold        = config.ecs_thres,
        explicit_zero_prob   = config.explicit_zero_prob,
        use_fast_transformer = config.fast_transformer,
        pre_norm             = config.pre_norm,
        use_mod              = config.use_mod,
        ntokens_mod          = ntokens_mod if config.use_mod else None,
        vocab_mod            = vocab_mod if config.use_mod else None,
    )

    # 사전 학습 가중치 이식
    with torch.no_grad():
        pretrained_emb = model_dict["encoder.embedding.weight"][gene_ids_pretrained, :]
        model.encoder.embedding.weight.data[:len(pretrained_genes), :] = pretrained_emb
        model.encoder.enc_norm.weight.data = model_dict["encoder.enc_norm.weight"]

    model.to(device)
    return model


# ──────────────────────────────────────────────
# Step 5. 학습 루프
# ──────────────────────────────────────────────
def run_training(
    model, config, logger, save_dir, device,
    tokenized_train, tokenized_valid,
    train_batch_labels, valid_batch_labels,
    gene_ids, vocab, adata,
    criterion_gep_gepc, criterion_dab, criterion_cls,
    optimizer, scheduler, scaler,
    adata_sorted=None,
):
    best_val_loss    = float("inf")
    best_model       = None
    best_model_epoch = 0

    if USE_WANDB:
        define_wandb_metrcis()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        train_data_pt, valid_data_pt = prepare_data(
            tokenized_train    = tokenized_train,
            tokenized_valid    = tokenized_valid,
            train_batch_labels = train_batch_labels,
            valid_batch_labels = valid_batch_labels,
            config             = config,
            epoch              = epoch,
            sort_seq_batch     = config.per_seq_batch_sample,
        )

        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size             = config.batch_size,
            shuffle                = True,
            intra_domain_shuffle   = False,
            drop_last              = False,
            per_seq_batch_sample   = config.per_seq_batch_sample,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size             = config.batch_size,
            shuffle                = False,
            intra_domain_shuffle   = False,
            drop_last              = False,
            per_seq_batch_sample   = config.per_seq_batch_sample,
        )

        if config.do_train:
            train(
                model              = model,
                loader             = train_loader,
                vocab              = vocab,
                criterion_gep_gepc = criterion_gep_gepc,
                criterion_dab      = criterion_dab,
                criterion_cls      = criterion_cls,
                scaler             = scaler,
                optimizer          = optimizer,
                scheduler          = scheduler,
                device             = device,
                config             = config,
                logger             = logger,
                epoch              = epoch,
            )

        val_loss = evaluate(
            model              = model,
            loader             = valid_loader,
            vocab              = vocab,
            criterion_gep_gepc = criterion_gep_gepc,
            criterion_dab      = criterion_dab,
            criterion_cls      = criterion_cls,
            device             = device,
            config             = config,
            epoch              = epoch,
        )

        elapsed = time.time() - epoch_start
        logger.info("-" * 89)
        logger.info(
            f"| epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model       = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model: val_loss={best_val_loss:5.4f}")

        # 주기적 평가 및 저장
        if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
            ckpt_path = save_dir / f"model_e{best_model_epoch}.pt"
            logger.info(f"모델 저장: {ckpt_path}")
            torch.save(best_model.state_dict(), ckpt_path)

            results = eval_testdata(
                model        = best_model,
                adata_t      = adata_sorted if config.per_seq_batch_sample else adata,
                gene_ids     = gene_ids,
                vocab        = vocab,
                config       = config,
                logger       = logger,
                include_types= ["cls"],
            )

            batch_umap_path = save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png"
            cell_umap_path  = save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png"
            results["batch_umap"].savefig(batch_umap_path, dpi=300)
            results["celltype_umap"].savefig(cell_umap_path, dpi=300)
            plt.close("all")

            if USE_WANDB:
                import wandb
                metrics = {"test/" + k: v for k, v in results.items()}
                metrics["test/batch_umap"]     = wandb.Image(str(batch_umap_path),
                                                              caption=f"batch umap e{best_model_epoch}")
                metrics["test/celltype_umap"]  = wandb.Image(str(cell_umap_path),
                                                              caption=f"celltype umap e{best_model_epoch}")
                metrics["test/best_model_epoch"] = best_model_epoch
                wandb.log(metrics)
                wandb.log({"avg_bio": results.get("avg_bio", 0.0)})

        scheduler.step()

    return best_model, best_model_epoch


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    # ── config 초기화 ──
    config, run = get_config()
    set_seed(config.seed)

    special_tokens = [config.pad_token, "<cls>", "<eoc>"]

    # ── 저장 디렉토리 및 로거 ──
    save_dir = Path(f"./results/multiomics_finetune/dev_{config.dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(f"저장 경로: {save_dir}")

    # ── 데이터 로드 ──
    adata, adata_protein, data_is_raw = load_bmmc(config)

    # ── 모달리티 정보 ──
    gene_loc_df, vocab_mod = build_modality_info(adata, adata_protein, special_tokens)
    ntokens_mod = len(vocab_mod)

    # ── vocab 로드 ──
    old_vocab, model_file = load_vocab(config, adata, special_tokens, logger)

    # ── 전처리 ──
    adata         = preprocess_rna(adata, config, data_is_raw)
    adata_protein = preprocess_protein(adata_protein, config)
    adata         = combine_modalities(adata, adata_protein)

    if config.per_seq_batch_sample:
        adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()
    else:
        adata_sorted = None

    # ── 학습 데이터 준비 ──
    all_counts = (
        adata.layers[config.input_layer_key].toarray()
        if issparse(adata.layers[config.input_layer_key])
        else adata.layers[config.input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()

    celltypes_labels = np.array(adata.obs["celltype"].tolist())
    num_types        = len(set(celltypes_labels))
    batch_ids        = np.array(adata.obs["batch_id"].tolist())
    num_batch_types  = len(set(batch_ids))

    # 모달리티 토큰
    mod_type = np.array([gene_loc_df.loc[g, "mod"] for g in genes])
    mod_type = np.array(vocab_mod(list(mod_type)), dtype=int)

    # train/valid 분리
    (train_data, valid_data,
     train_celltype_labels, valid_celltype_labels,
     train_batch_labels, valid_batch_labels) = train_test_split(
        all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
    )

    # 데이터 통계
    nz = [np.count_nonzero(train_data[i]) for i in range(train_data.shape[0])]
    logger.info(f"non-zero 유전자 - max: {np.max(nz)}, min: {np.min(nz)}, mean: {np.mean(nz):.1f}")
    logger.info(f"세포 유형 수: {num_types}")

    # ── vocab / gene_ids 구성 ──
    vocab, gene_ids, pretrained_genes, gene_ids_pretrained = build_gene_vocab(
        genes, special_tokens, old_vocab
    )

    # ── 토크나이징 ──
    tokenize_kwargs = dict(
        max_len           = config.max_seq_len,
        vocab             = vocab,
        pad_token         = config.pad_token,
        pad_value         = config.pad_value,
        append_cls        = True,
        include_zero_gene = config.include_zero_gene,
        mod_type          = mod_type if config.use_mod else None,
        vocab_mod         = vocab_mod if config.use_mod else None,
    )
    tokenized_train = tokenize_and_pad_batch(train_data, gene_ids, **tokenize_kwargs)
    tokenized_valid = tokenize_and_pad_batch(valid_data, gene_ids, **tokenize_kwargs)
    logger.info(
        f"train: {tokenized_train['genes'].shape[0]} samples, "
        f"feature len: {tokenized_train['genes'].shape[1]}"
    )

    # ── 모델 구성 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(
        config, vocab, model_file, device,
        num_batch_types, ntokens_mod, vocab_mod,
        pretrained_genes, gene_ids_pretrained,
    )
    logger.info(str(model))

    if USE_WANDB:
        wandb.watch(model)

    # ── 손실 함수 / 옵티마이저 ──
    criterion_gep_gepc = masked_mse_loss        if config.GEP and config.GEPC else None
    criterion_cls      = nn.CrossEntropyLoss()  if config.CLS                  else None
    criterion_dab      = nn.CrossEntropyLoss()  if config.DAR                  else None

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr,
        eps=1e-4 if config.amp else 1e-8,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
    scaler    = torch.cuda.amp.GradScaler(enabled=config.amp)

    # ── 학습 ──
    best_model, best_model_epoch = run_training(
        model              = model,
        config             = config,
        logger             = logger,
        save_dir           = save_dir,
        device             = device,
        tokenized_train    = tokenized_train,
        tokenized_valid    = tokenized_valid,
        train_batch_labels = train_batch_labels,
        valid_batch_labels = valid_batch_labels,
        gene_ids           = gene_ids,
        vocab              = vocab,
        adata              = adata,
        criterion_gep_gepc = criterion_gep_gepc,
        criterion_dab      = criterion_dab,
        criterion_cls      = criterion_cls,
        optimizer          = optimizer,
        scheduler          = scheduler,
        scaler             = scaler,
        adata_sorted       = adata_sorted,
    )

    # ── 최종 모델 저장 ──
    best_model_path = save_dir / "best_model.pt"
    torch.save(best_model.state_dict(), best_model_path)
    logger.info(f"최종 모델 저장: {best_model_path}")

    # ── W&B 아티팩트 등록 및 종료 ──
    if USE_WANDB and run is not None:
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(str(best_model_path))
        run.log_artifact(artifact)
        run.finish()
        wandb.finish()

    gc.collect()
    logger.info("완료.")


if __name__ == "__main__":
    main()
