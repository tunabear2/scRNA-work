"""
07. Identifying Celltype-specific lncRNAs (PyTorch + CellFM-torch)
CellFM 인코더의 cls_token ↔ 유전자 임베딩 유사도로 세포 유형별 고-주목 유전자를 식별.

데이터: datasets/CellFM/PBMC_10K.h5ad
모델:   checkpoint/CellFM_80M_weight.ckpt (사전학습 가중치)
출력:  results/07_lncrna/lncrna_results.csv
       results/07_lncrna/metrics.json
"""
import os, sys, datetime
import numpy as np
import pandas as pd
import scanpy as sc
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR   = os.path.join(ROOT_DIR, "scripts")
CELLFM_TORCH = os.path.join(ROOT_DIR, "CellFM-torch")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CELLFM_TORCH)

os.chdir(ROOT_DIR)

from log_utils.logger import get_logger, save_metrics
from layers.utils import Config_80M, Prepare
from model import Cell_FM

TASK      = "07_lncrna"
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "CellFM", "PBMC_10K.h5ad")
CSV_DIR   = os.path.join(ROOT_DIR, "csv")
RESULTS   = os.path.join(ROOT_DIR, "results", TASK)
FIG_DIR   = os.path.join(RESULTS, "figures")
PRETRAIN  = os.path.join(ROOT_DIR, "checkpoint", "CellFM_80M_weight.ckpt")

os.makedirs(FIG_DIR, exist_ok=True)
logger = get_logger(TASK, RESULTS)

DEVICE = "cuda:0"
TOPK   = 100


class LncRNADataset(Dataset):
    """PBMC 데이터를 CellFM 입력 형식으로 변환."""

    def __init__(self, adata, gene_info):
        from scipy.sparse import csr_matrix as csm
        geneset = {j: i + 1 for i, j in enumerate(gene_info.index)}

        # normalize
        data = adata.X.astype(np.float32)
        T    = adata.X.sum(1)
        data = csm(np.round(data / np.maximum(1, np.asarray(T) / 1e5, dtype=np.float32)))
        data.eliminate_zeros()

        # map genes (최대 2048)
        used_genes = [g for g in adata.var_names if g in geneset]
        used_genes = used_genes[:2048]
        pad_num    = 2048 - len(used_genes)

        cols  = [list(adata.var_names).index(g) for g in used_genes]
        X     = data[:, cols].toarray().astype(np.int32)

        self.gene   = np.array([geneset[g] for g in used_genes] + [0]*pad_num, dtype=np.int32)
        self.data   = X
        self.T      = np.asarray(data.sum(1)).ravel()
        self.labels = adata.obs["cell_type"].astype("category").cat.codes.values
        self.id2type= list(adata.obs["cell_type"].astype("category").cat.categories)
        self.gene_names = used_genes  # non-padded gene names
        logger.info(f"LncRNADataset: {len(adata)} cells, {len(used_genes)} genes")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].astype(np.float32), self.gene.copy(), self.T[idx], self.labels[idx]


def make_loader(dataset, prep, batch_size=16):
    def collate(samples):
        raw_b, dw_b, ST_b, ng_b, mg_b, zi_b, label_b = [], [], [], [], [], [], []
        for data, gene, T, label in samples:
            raw_data, nonz, zero = prep.seperate(data)
            data, nonz, _, _, seq_len = prep.sample(raw_data, nonz, zero)
            raw_data, raw_nzdata, nonz = prep.compress(raw_data, nonz)
            gene_arr, nonz_gene, _    = prep.compress(gene, nonz)

            raw_nzdata, dw_nzdata, S, T_v = prep.bayes(raw_nzdata, T)
            dw_nzdata, S   = prep.normalize(dw_nzdata, S)
            raw_nzdata, T_ = prep.normalize(raw_nzdata, T_v)
            ST_feat        = prep.cat_st(S, T_v)

            zero_idx  = prep.attn_mask(seq_len)
            dw_nzdata, mask_gene = prep.mask(dw_nzdata)

            raw_nzdata = prep.pad_zero(raw_nzdata)
            dw_nzdata  = prep.pad_zero(dw_nzdata)
            nonz_gene  = prep.pad_zero(nonz_gene)
            mask_gene  = prep.pad_zero(mask_gene)

            raw_b.append(torch.tensor(raw_nzdata, dtype=torch.float32))
            dw_b.append(torch.tensor(dw_nzdata,   dtype=torch.float32))
            ST_b.append(torch.tensor(ST_feat,      dtype=torch.float32))
            ng_b.append(torch.tensor(nonz_gene,    dtype=torch.int32))
            mg_b.append(torch.tensor(mask_gene,    dtype=torch.float32))
            zi_b.append(torch.tensor(zero_idx,     dtype=torch.float32))
            label_b.append(label)

        return (
            torch.stack(raw_b), torch.stack(dw_b), torch.stack(ST_b),
            torch.stack(ng_b),  torch.stack(mg_b),  torch.stack(zi_b),
            torch.tensor(label_b, dtype=torch.long),
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      drop_last=False, collate_fn=collate, num_workers=2)


def analyze_attention(net, loader, dataset, num_cls):
    """
    cls_token ↔ 각 유전자 임베딩 내적(dot-product) 로 attention 대리 점수 계산.
    emb[:, 0] = cls_token, emb[:, 3:] = gene expr embeddings.
    """
    attn_map = {i: defaultdict(list) for i in range(num_cls)}
    gene_names = dataset.gene_names  # non-padded

    net.eval()
    with torch.no_grad():
        for raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx, labels in tqdm(loader, desc="[07] 어텐션 분석"):
            raw_nzdata = raw_nzdata.to(DEVICE)
            dw_nzdata  = dw_nzdata.to(DEVICE)
            ST_feat    = ST_feat.to(DEVICE)
            nonz_gene  = nonz_gene.to(DEVICE)
            zero_idx   = zero_idx.to(DEVICE)

            with torch.cuda.amp.autocast():
                emb, gene_emb = net.net.encode(dw_nzdata, nonz_gene, ST_feat, zero_idx)

            # emb: [B, 3+L, D] — [cls, st1, st2, gene0, gene1, ...]
            cls_token = emb[:, 0]          # [B, D]
            expr_emb  = emb[:, 3:]         # [B, L, D]

            # dot-product: [B, L]
            scores = torch.bmm(
                cls_token.unsqueeze(1).float(),
                expr_emb.permute(0, 2, 1).float()
            ).squeeze(1).cpu().numpy()     # [B, L]

            nonz_gene_np = nonz_gene.cpu().numpy()  # [B, L_pad]
            L_used = min(expr_emb.shape[1], len(gene_names))

            for b in range(len(labels)):
                ct = labels[b].item()
                for pos in range(L_used):
                    gene_idx = nonz_gene_np[b, pos]
                    if gene_idx > 0 and pos < len(gene_names):
                        attn_map[ct][gene_names[pos]].append(float(scores[b, pos]))

    # 평균 집계 및 정렬
    result = {}
    for ct in range(num_cls):
        avg_scores = {g: np.mean(v) for g, v in attn_map[ct].items() if v}
        result[ct]  = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    return result


def main():
    adata = sc.read_h5ad(DATA_PATH)
    logger.info(f"데이터 로드: {adata.shape}, obs={list(adata.obs.columns)}")

    # cell_type 컬럼 확인
    if "cell_type" not in adata.obs.columns:
        if "str_labels" in adata.obs.columns:
            adata.obs["cell_type"] = adata.obs["str_labels"]
        else:
            raise KeyError("cell_type 컬럼 없음")

    gene_info = pd.read_csv(os.path.join(CSV_DIR, "expand_gene_info.csv"),
                            index_col=0, header=0)

    dataset  = LncRNADataset(adata, gene_info)
    num_cls  = len(dataset.id2type)

    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs      = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.ckpt_path= PRETRAIN
    cfg.device   = DEVICE

    prep   = Prepare(cfg.nonz_len, pad=0, mask_ratio=0.0, dw=False)
    loader = make_loader(dataset, prep, batch_size=16)

    net = Cell_FM(27855, cfg, ckpt_path=PRETRAIN, device=DEVICE).to(DEVICE)
    net.load_model(weight=True, moment=False)
    logger.info("사전학습 모델 로드 완료")

    # geneset 역방향 매핑 (gene_idx → gene_name, feature_type)
    feature_map = dict(zip(gene_info.index, gene_info["feature"])) \
        if "feature" in gene_info.columns else {}

    logger.info("어텐션 분석 시작...")
    attention = analyze_attention(net, loader, dataset, num_cls)

    # ── 결과 저장 ────────────────────────────────────────────
    rows = []
    summary = {}
    for ct_idx in range(num_cls):
        ct_label  = dataset.id2type[ct_idx]
        top_genes = attention[ct_idx][:TOPK]

        lncrna_genes  = [g for g, _ in top_genes
                         if feature_map.get(g, "protein coding") != "protein coding"]
        n_lncrna      = len(lncrna_genes)
        summary[ct_label] = {
            "top100_lncRNA_count": n_lncrna,
            "top100_lncRNA_ratio": round(n_lncrna / TOPK, 3),
        }
        logger.info(f"{ct_label}: lncRNA top5={lncrna_genes[:5]}")

        for rank, (gene, score) in enumerate(top_genes):
            rows.append({
                "cell_type": ct_label,
                "rank":      rank + 1,
                "gene":      gene,
                "attn_score":score,
                "gene_type": feature_map.get(gene, "unknown"),
            })

    df_out   = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS, "lncrna_results.csv")
    df_out.to_csv(csv_path, index=False)
    logger.info(f"lncRNA 결과 저장: {csv_path}")

    metrics = {
        "task":      TASK,
        "dataset":   "PBMC_10K",
        "model":     PRETRAIN,
        "topk":      TOPK,
        "summary":   summary,
        "timestamp": datetime.datetime.now().isoformat(),
        "output":    csv_path,
    }
    path = save_metrics(metrics, RESULTS)
    logger.info(f"결과 저장: {path}")


if __name__ == "__main__":
    main()
