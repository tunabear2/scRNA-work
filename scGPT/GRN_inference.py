"""
scGPT Gene Regulatory Network (GRN) Inference Script

Steps:
    1. Load pre-trained scGPT model and dataset
    2. Retrieve gene embeddings
    3. Extract gene programs via Louvain clustering
    4. Visualize gene program activations
    5. Visualize gene network connectivity
    6. Reactome pathway enrichment analysis

Requirements:
    - Pre-trained blood model: https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU
    - Immune Human dataset:    https://figshare.com/ndownloader/files/25717328
"""

import copy
import json
import os
import sys
import warnings
from pathlib import Path

import torch
import scanpy as sc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import tqdm
import gseapy as gp
from anndata import AnnData
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

# scGPT 경로 설정 (필요 시 수정)
sys.path.insert(0, "../")
import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────
MODEL_DIR   = Path("./data/models/pretrain_bc")     # 모델 경로
DATA_PATH   = Path("./data/GRN_inference/Immune_ALL_human.h5ad")   # 데이터 경로
OUTPUT_DIR  = Path("./results/GRN_inference")  # 결과 저장 경로

PAD_TOKEN   = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
N_HVG       = 1200
N_BINS      = 51
MASK_VALUE  = -1
PAD_VALUE   = -2

LOUVAIN_RESOLUTION = 40   # Louvain 클러스터링 해상도
MIN_GENES_PER_PROGRAM = 5  # 유전자 프로그램 최소 유전자 수

NETWORK_SIMILARITY_THRESH = 0.4  # 네트워크 시각화 엣지 임계값
TARGET_GENE_PROGRAM = "3"        # 시각화할 유전자 프로그램 ID

REACTOME_DB = ["Reactome_2022"]
PVALUE_THRESH = 0.05

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Step 1. 모델 및 데이터 로드
# ──────────────────────────────────────────────
def load_model(model_dir: Path):
    model_config_file = model_dir / "args.json"
    model_file        = model_dir / "best_model.pt"
    vocab_file        = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            vocab.append_token(token)

    with open(model_config_file) as f:
        cfg = json.load(f)

    print(f"모델 로드: {model_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(
        ntoken    = len(vocab),
        d_model   = cfg["embsize"],
        nhead     = cfg["nheads"],
        d_hid     = cfg["d_hid"],
        nlayers   = cfg["nlayers"],
        vocab     = vocab,
        pad_value = PAD_VALUE,
        n_input_bins = N_BINS,
    )

    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
    except Exception:
        state = model.state_dict()
        pretrained = torch.load(model_file, map_location=device)
        pretrained = {k: v for k, v in pretrained.items()
                      if k in state and v.shape == state[k].shape}
        state.update(pretrained)
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model, vocab, device


def load_data(data_path: Path):
    adata = sc.read(str(data_path), cache=True)
    adata.obs["celltype"] = adata.obs["final_annotation"].astype(str)
    return adata


def preprocess_data(adata, data_is_raw: bool = False):
    preprocessor = Preprocessor(
        use_key               = "X",
        filter_gene_by_counts = 3,
        filter_cell_by_counts = False,
        normalize_total       = 1e4,
        result_normed_key     = "X_normed",
        log1p                 = data_is_raw,
        result_log1p_key      = "X_log1p",
        subset_hvg            = N_HVG,
        hvg_flavor            = "seurat_v3" if data_is_raw else "cell_ranger",
        binning               = N_BINS,
        result_binned_key     = "X_binned",
    )
    preprocessor(adata, batch_key="batch")
    return adata


# ──────────────────────────────────────────────
# Step 2. 유전자 임베딩 추출
# ──────────────────────────────────────────────
def get_gene_embeddings(model, vocab, device, adata):
    gene2idx = vocab.get_stoi()
    gene_ids = np.array(list(gene2idx.values()))

    with torch.no_grad():
        embeddings = model.encoder(
            torch.tensor(gene_ids, dtype=torch.long, device=device)
        ).cpu().numpy()

    hvg_genes = set(adata.var.index.tolist())
    gene_embeddings = {
        gene: embeddings[i]
        for i, gene in enumerate(gene2idx.keys())
        if gene in hvg_genes
    }
    print(f"유전자 임베딩 수: {len(gene_embeddings)}")
    return gene_embeddings


# ──────────────────────────────────────────────
# Step 3. 유전자 프로그램 추출 (Louvain 클러스터링)
# ──────────────────────────────────────────────
def extract_gene_programs(gene_embeddings, resolution=40, min_genes=5):
    embed   = GeneEmbedding(gene_embeddings)
    gdata   = embed.get_adata(resolution=resolution)
    metagenes = embed.get_metagenes(gdata)

    gene_programs = {
        mg: genes
        for mg, genes in metagenes.items()
        if len(genes) >= min_genes
    }
    print(f"유전자 프로그램 수 (≥{min_genes}개 유전자): {len(gene_programs)}")
    return embed, gene_programs, metagenes


# ──────────────────────────────────────────────
# Step 4. 유전자 프로그램 활성화 시각화
# ──────────────────────────────────────────────
def visualize_program_activation(embed, adata, metagenes, gene_programs):
    sns.set(font_scale=0.35)
    embed.score_metagenes(adata, metagenes)
    embed.plot_metagenes_scores(adata, gene_programs, "celltype")

    out_path = OUTPUT_DIR / "gene_program_activation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"활성화 시각화 저장: {out_path}")


# ──────────────────────────────────────────────
# Step 5. 유전자 네트워크 시각화
# ──────────────────────────────────────────────
def visualize_gene_network(embed, gene_programs, program_id, threshold=0.4):
    genes = gene_programs[program_id]
    print(f"프로그램 {program_id} 유전자: {genes}")

    rows = []
    for gene in tqdm.tqdm(genes, desc="cosine similarity 계산"):
        df_sim = embed.compute_similarities(gene, genes)
        df_sim["Gene1"] = gene
        rows.append(df_sim)

    df_all = pd.concat(rows, ignore_index=True)
    df_sub = df_all[df_all["Similarity"] < 0.99].sort_values("Gene")

    edges = [
        (row["Gene"], row["Gene1"], round(row["Similarity"], 2))
        for _, row in df_sub.iterrows()
    ]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    pos = nx.spring_layout(G, k=0.4, iterations=15, seed=3)
    widths = nx.get_edge_attributes(G, "weight")

    large_edges = {e: w * 10 for e, w in widths.items() if w > threshold}
    small_edges  = {e: max(w, 0) * 10 for e, w in widths.items() if w <= threshold}

    plt.figure(figsize=(20, 20))
    nx.draw_networkx_edges(G, pos, edgelist=small_edges.keys(),
                           width=list(small_edges.values()),
                           edge_color="lightblue", alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=large_edges.keys(),
                           width=list(large_edges.values()),
                           alpha=0.5, edge_color="blue")
    nx.draw_networkx_labels(G, pos, font_size=25, font_family="sans-serif")

    edge_labels = {e: widths[e] for e in large_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)

    plt.gca().margins(0.08)
    plt.axis("off")
    out_path = OUTPUT_DIR / f"gene_network_program_{program_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"네트워크 시각화 저장: {out_path}")

    return genes


# ──────────────────────────────────────────────
# Step 6. Reactome 경로 농축 분석
# ──────────────────────────────────────────────
DB_TERM_COUNTS = {
    "GO_Biological_Process_2021": 6036,
    "GO_Molecular_Function_2021": 1274,
    "Reactome_2022": 1818,
}

def run_pathway_analysis(gene_list, databases=None, p_thresh_base=0.05):
    if databases is None:
        databases = REACTOME_DB

    total_terms = sum(DB_TERM_COUNTS.get(db, 0) for db in databases)
    corrected_p = p_thresh_base / total_terms

    try:
        enr = gp.enrichr(
            gene_list  = gene_list,
            gene_sets  = databases,
            organism   = "human",
            outdir     = str(OUTPUT_DIR / "enrichr"),
            cutoff     = 0.5,
        )
        results = enr.results
        results = results[results["P-value"] < corrected_p].reset_index(drop=True)

        out_path = OUTPUT_DIR / "pathway_enrichment.csv"
        results.to_csv(out_path, index=False)
        print(f"경로 분석 결과 저장: {out_path}")
        print(results)
        return results
    except Exception as e:
        print(f"[WARNING] Enrichr API call failed (network issue): {e}")
        print("Skipping Reactome pathway validation. Core GRN results are unaffected.")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    set_seed(42)

    # 1. 모델 & 데이터 로드
    model, vocab, device = load_model(MODEL_DIR)
    adata = load_data(DATA_PATH)
    adata = preprocess_data(adata, data_is_raw=False)

    # 2. 유전자 임베딩 추출
    gene_embeddings = get_gene_embeddings(model, vocab, device, adata)

    # 3. 유전자 프로그램 추출
    embed, gene_programs, metagenes = extract_gene_programs(
        gene_embeddings,
        resolution=LOUVAIN_RESOLUTION,
        min_genes=MIN_GENES_PER_PROGRAM,
    )

    # 4. 활성화 시각화
    visualize_program_activation(embed, adata, metagenes, gene_programs)

    # 5. 네트워크 시각화 (유전자 프로그램 3번 예시)
    target_genes = visualize_gene_network(
        embed, gene_programs,
        program_id=TARGET_GENE_PROGRAM,
        threshold=NETWORK_SIMILARITY_THRESH,
    )

    # 6. 경로 분석
    run_pathway_analysis(target_genes, databases=REACTOME_DB)


if __name__ == "__main__":
    main()
