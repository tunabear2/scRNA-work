"""
03. Batch Effect Integration (PyTorch + CellFM-torch)
사전학습 CellFM으로 임베딩 추출 → scib 메트릭 계산 → UMAP 시각화.
데이터: datasets/CellFM/PBMC_10K.h5ad
출력:  results/03_batch_integration/  (embedding, UMAP, metrics.json)
"""
import os, sys, datetime
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from tqdm import tqdm
from scipy.sparse import issparse

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR   = os.path.join(ROOT_DIR, "scripts")
CELLFM_TORCH = os.path.join(ROOT_DIR, "CellFM-torch")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CELLFM_TORCH)

os.chdir(ROOT_DIR)

from log_utils.logger import get_logger, save_metrics
from layers.utils import Config_80M, Prepare
from layers.torch_finetune import FinetuneModel
from model import Cell_FM

TASK      = "03_batch_integration"
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "CellFM", "PBMC_10K.h5ad")
CSV_DIR   = os.path.join(ROOT_DIR, "csv")
RESULTS   = os.path.join(ROOT_DIR, "results", TASK)
FIG_DIR   = os.path.join(RESULTS, "figures")
PRETRAIN  = os.path.join(ROOT_DIR, "checkpoint", "CellFM_80M_weight.ckpt")

os.makedirs(FIG_DIR, exist_ok=True)
logger = get_logger(TASK, RESULTS)

DEVICE = "cuda:0"
BATCH  = 32


# ── 데이터셋 ───────────────────────────────────────────────────
class PBMCDataset(Dataset):
    def __init__(self, adata, gene_info):
        from scipy.sparse import csr_matrix as csm
        geneset = {j: i + 1 for i, j in enumerate(gene_info.index)}

        # normalize X
        data = adata.X.astype(np.float32)
        T    = adata.X.sum(1)
        data = csm(np.round(data / np.maximum(1, np.asarray(T) / 1e5, dtype=np.float32)))
        data.eliminate_zeros()

        # select & map genes
        used_genes = [g for g in adata.var_names if g in geneset]
        used_genes = used_genes[:2048]
        pad_num    = 2048 - len(used_genes)

        X = data[:, [list(adata.var_names).index(g) for g in used_genes]]
        X = X.toarray().astype(np.int32)

        self.gene   = np.array(
            [geneset[g] for g in used_genes] + [0] * pad_num, dtype=np.int32)
        self.data   = X
        self.T      = np.asarray(data.sum(1)).ravel()
        self.labels = adata.obs["cell_type"].astype("category").cat.codes.values
        self.batches= adata.obs["batch"].values
        self.n_sel  = len(used_genes)
        logger.info(f"PBMC Dataset: {len(adata)} cells, {len(used_genes)} genes used")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].astype(np.float32)
        return data, self.gene.copy(), self.T[idx], self.labels[idx], self.batches[idx]


def make_loader(dataset, prep, batch_size):
    def collate(samples):
        raw_nzdata_b, dw_nzdata_b, ST_feat_b = [], [], []
        nonz_gene_b, mask_gene_b, zero_idx_b = [], [], []
        label_b, batch_b = [], []

        for data, gene, T, label, batch_id in samples:
            raw_data, nonz, zero = prep.seperate(data)
            data, nonz, _, _, seq_len = prep.sample(raw_data, nonz, zero)
            raw_data, raw_nzdata, nonz = prep.compress(raw_data, nonz)
            gene_arr, nonz_gene, _ = prep.compress(gene, nonz)

            raw_nzdata, dw_nzdata, S, T_val = prep.bayes(raw_nzdata, T)
            dw_nzdata, S   = prep.normalize(dw_nzdata, S)
            raw_nzdata, T_ = prep.normalize(raw_nzdata, T_val)
            ST_feat        = prep.cat_st(S, T_val)

            zero_idx  = prep.attn_mask(seq_len)
            dw_nzdata, mask_gene = prep.mask(dw_nzdata)

            raw_nzdata = prep.pad_zero(raw_nzdata)
            dw_nzdata  = prep.pad_zero(dw_nzdata)
            nonz_gene  = prep.pad_zero(nonz_gene)
            mask_gene  = prep.pad_zero(mask_gene)

            raw_nzdata_b.append(torch.tensor(raw_nzdata, dtype=torch.float32))
            dw_nzdata_b.append(torch.tensor(dw_nzdata,  dtype=torch.float32))
            ST_feat_b.append(torch.tensor(ST_feat,       dtype=torch.float32))
            nonz_gene_b.append(torch.tensor(nonz_gene,  dtype=torch.int32))
            mask_gene_b.append(torch.tensor(mask_gene,  dtype=torch.float32))
            zero_idx_b.append(torch.tensor(zero_idx,    dtype=torch.float32))
            label_b.append(label)
            batch_b.append(batch_id)

        return {
            "raw_nzdata": torch.stack(raw_nzdata_b),
            "dw_nzdata":  torch.stack(dw_nzdata_b),
            "ST_feat":    torch.stack(ST_feat_b),
            "nonz_gene":  torch.stack(nonz_gene_b),
            "mask_gene":  torch.stack(mask_gene_b),
            "zero_idx":   torch.stack(zero_idx_b),
            "label":      torch.tensor(label_b, dtype=torch.long),
            "batch_id":   torch.tensor(batch_b, dtype=torch.long),
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      drop_last=False, collate_fn=collate, num_workers=2)


# ── scib 메트릭 ───────────────────────────────────────────────
def compute_scib(adata_emb):
    try:
        import scib
        metric = scib.metrics.metrics(
            adata_emb, adata_emb,
            batch_key="batch", label_key="cell_type",
            embed="X_emb",
            ari_=True, nmi_=True, silhouette_=True, isolated_labels_asw_=True,
            pcr_=False, kBET_=False, graph_conn_=False, trajectory_=False,
        )
        return metric.to_dict() if hasattr(metric, "to_dict") else dict(metric)
    except Exception as e:
        logger.warning(f"scib 메트릭 계산 실패: {e}")
        return {}


def main():
    # ── 데이터 로드 ──────────────────────────────────────────
    adata = sc.read_h5ad(DATA_PATH)
    logger.info(f"데이터 로드: {adata.shape}, obs={list(adata.obs.columns)}")

    gene_info = pd.read_csv(os.path.join(CSV_DIR, "expand_gene_info.csv"),
                            index_col=0, header=0)

    dataset = PBMCDataset(adata, gene_info)
    cfg     = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs      = True
    cfg.add_zero = True
    cfg.pad_zero = True

    prep   = Prepare(cfg.nonz_len, pad=0, mask_ratio=0.0, dw=False)
    loader = make_loader(dataset, prep, BATCH)

    # ── 모델 로드 ────────────────────────────────────────────
    cfg.ckpt_path = PRETRAIN
    cfg.device    = DEVICE
    net = Cell_FM(27855, cfg, ckpt_path=PRETRAIN, device=DEVICE).to(DEVICE)
    net.load_model(weight=True, moment=False)
    net.eval()
    logger.info("사전학습 모델 로드 완료 (임베딩 추출 전용)")

    # ── 임베딩 추출 ──────────────────────────────────────────
    all_embs   = []
    all_labels = []
    all_batches= []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[03] 임베딩 추출"):
            raw_nzdata = batch["raw_nzdata"].to(DEVICE)
            dw_nzdata  = batch["dw_nzdata"].to(DEVICE)
            ST_feat    = batch["ST_feat"].to(DEVICE)
            nonz_gene  = batch["nonz_gene"].to(DEVICE)
            zero_idx   = batch["zero_idx"].to(DEVICE)

            with torch.amp.autocast("cuda"):
                emb, _ = net.net.encode(dw_nzdata, nonz_gene, ST_feat, zero_idx)
            cls_token = emb[:, 0].cpu().float().numpy()

            all_embs.append(cls_token)
            all_labels.extend(batch["label"].numpy())
            all_batches.extend(batch["batch_id"].numpy())

    embeddings = np.concatenate(all_embs, axis=0)
    logger.info(f"임베딩 추출 완료: shape={embeddings.shape}")

    # numpy 저장
    np.save(os.path.join(RESULTS, "PBMC_embeddings.npy"), embeddings)

    # ── scib 메트릭 ──────────────────────────────────────────
    id2label = adata.obs["cell_type"].astype("category").cat.categories.tolist()
    adata_emb = sc.AnnData(embeddings)
    adata_emb.obs["cell_type"] = [id2label[l] for l in all_labels]
    adata_emb.obs["batch"]     = [str(b) for b in all_batches]
    adata_emb.obsm["X_emb"]   = embeddings

    metric = compute_scib(adata_emb)
    logger.info(f"scib 메트릭: {metric}")

    # ── UMAP ─────────────────────────────────────────────────
    sc.pp.neighbors(adata_emb, use_rep="X_emb", n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_emb, min_dist=0.3)

    for color_key in ["cell_type", "batch"]:
        fig = sc.pl.umap(adata_emb, color=[color_key], frameon=False,
                         return_fig=True, show=False)
        fig_path = os.path.join(FIG_DIR, f"umap_{color_key}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"UMAP 저장: {fig_path}")

    # ── 통합 임베딩 h5ad 저장 ─────────────────────────────────────
    h5ad_path = os.path.join(RESULTS, "PBMC_integrated.h5ad")
    adata_emb.write_h5ad(h5ad_path)
    logger.info(f"통합 임베딩 h5ad 저장: {h5ad_path}")

    metrics = {
        "task":       TASK,
        "dataset":    "PBMC_10K",
        "scib":       metric,
        "timestamp":  datetime.datetime.now().isoformat(),
        "embedding":  str(os.path.join(RESULTS, "PBMC_embeddings.npy")),
        "h5ad":       h5ad_path,
    }
    path = save_metrics(metrics, RESULTS)
    logger.info(f"결과 저장: {path}")


if __name__ == "__main__":
    main()
