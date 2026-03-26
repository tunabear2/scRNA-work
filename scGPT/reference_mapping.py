#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT Reference Mapping
- Mode 1: 커스텀 레퍼런스 데이터셋으로 매핑 (faiss 없이 실행 가능)
- Mode 2: CellXGene 아틀라스로 매핑       (faiss 필수)

[Mode 2 faiss 설치]
  conda: conda install -c pytorch faiss-cpu   # CPU 버전
         conda install -c pytorch faiss-gpu   # GPU 버전
  참고:  https://github.com/facebookresearch/faiss/wiki/Installing-Faiss

[Mode 2 인덱스 다운로드]
  https://drive.google.com/drive/folders/1q14U50SNg5LMjlZ9KH-n-YsGRi8zkCbe?usp=sharing
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import sklearn.metrics
import scanpy as sc
from tqdm import tqdm

sys.path.insert(0, "../")
import scgpt as scg

warnings.filterwarnings("ignore", category=ResourceWarning)

# faiss 설치 여부 확인
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ──────────────────────────────────────────────
# 공통 설정
# ──────────────────────────────────────────────
MODEL_DIR = Path("../save/scGPT_human")
CELL_TYPE_KEY = "Celltype"
GENE_COL = "index"
K_CUSTOM = 10   # 커스텀 레퍼런스 이웃 수
K_ATLAS  = 50   # CellXGene 아틀라스 이웃 수
SAVE_DIR = Path("./results/reference_mapping")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Mode 1 전용: faiss fallback 유사도 함수
# ──────────────────────────────────────────────
def l2_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return -np.linalg.norm(a - b, axis=1)

def get_similar_vectors(vector: np.ndarray, ref: np.ndarray, top_k: int = 10):
    sims = l2_sim(vector, ref)
    top_k_idx = np.argsort(sims)[::-1][:top_k]
    return top_k_idx, sims[top_k_idx]


# ──────────────────────────────────────────────
# Mode 2 전용: faiss 인덱스 로드 & 투표 함수
# (build_atlas_index_faiss.py 에서 필요한 부분만 내장)
# ──────────────────────────────────────────────
def _auto_set_nprobe(index, nprobe: Optional[int] = None) -> Optional[int]:
    """IVF 인덱스의 nprobe를 클러스터 수에 맞게 자동 설정."""
    index_ivf = faiss.try_extract_index_ivf(index)
    if index_ivf:
        nlist = index_ivf.nlist
        index_ivf.nprobe = (
            nprobe if nprobe is not None
            else 16  if nlist <= 1e3
            else 32  if nlist <= 4e3
            else 64  if nlist <= 1.6e4
            else 128
        )
        print(f"nprobe 설정: {index_ivf.nprobe} (클러스터 수: {nlist})")
        return index_ivf.nprobe

def load_index(
    index_dir: str,
    use_config_file: bool = True,
    use_gpu: bool = False,
    nprobe: Optional[int] = None,
) -> Tuple:
    """faiss 인덱스와 메타 레이블을 디스크에서 로드."""
    index_file  = os.path.join(index_dir, "index.faiss")
    meta_file   = os.path.join(index_dir, "meta.h5ad")
    config_file = os.path.join(index_dir, "index_config.json")

    print(f"인덱스 로드 중: {index_dir}")
    index = faiss.read_index(index_file)
    print(f"로드 완료, 총 셀 수: {index.ntotal:,}")

    with h5py.File(meta_file, "r") as f:
        meta_labels = f["meta_labels"].asstr()[:]

    if use_config_file:
        with open(config_file, "r") as f:
            config = json.load(f)
        use_gpu = config["gpu"]
        nprobe  = config["nprobe"]

    _auto_set_nprobe(index, nprobe=nprobe)

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    return index, meta_labels

def vote(predicts_for_query: np.ndarray, return_prob: bool = True) -> Tuple:
    """다수결 투표로 셀 타입 예측."""
    unique_labels, counts = np.unique(predicts_for_query, return_counts=True)
    probs = counts / counts.sum()
    sorted_idx = np.argsort(probs)[::-1]
    if return_prob:
        return unique_labels[sorted_idx], probs[sorted_idx]
    return unique_labels[sorted_idx]


# ──────────────────────────────────────────────
# Mode 1: 커스텀 레퍼런스 데이터셋으로 매핑
# ──────────────────────────────────────────────
def run_custom_reference_mapping():
    print("\n=== Mode 1: 커스텀 레퍼런스 매핑 ===")

    # 레퍼런스 임베딩
    ref_adata = sc.read_h5ad("../data/annotation_pancreas/demo_train.h5ad")
    ref_embed = scg.tasks.embed_data(
        ref_adata, MODEL_DIR,
        gene_col=GENE_COL,
        obs_to_save=CELL_TYPE_KEY,
        batch_size=64,
        return_new_adata=True,
    )

    # 레퍼런스 UMAP 시각화
    sc.pp.neighbors(ref_embed, use_rep="X")
    sc.tl.umap(ref_embed)
    sc.pl.umap(ref_embed, color=CELL_TYPE_KEY, frameon=False, wspace=0.4)

    # 쿼리 임베딩
    test_adata = sc.read_h5ad("../data/annotation_pancreas/demo_test.h5ad")
    test_embed = scg.tasks.embed_data(
        test_adata, MODEL_DIR,
        gene_col=GENE_COL,
        obs_to_save=CELL_TYPE_KEY,
        batch_size=64,
        return_new_adata=True,
    )

    # KNN 레이블 전파 (faiss fallback: L2 거리 기반)
    ref_X  = ref_embed.X
    test_X = test_embed.X
    preds  = []
    for i in tqdm(range(test_X.shape[0]), desc="KNN 예측"):
        idx, _ = get_similar_vectors(test_X[i][np.newaxis, ...], ref_X, top_k=K_CUSTOM)
        top_label = ref_embed.obs[CELL_TYPE_KEY][idx].value_counts().index[0]
        preds.append(top_label)

    gt  = test_adata.obs[CELL_TYPE_KEY].to_numpy()
    acc = sklearn.metrics.accuracy_score(gt, preds)
    print(f"Accuracy: {acc:.4f}")

    # 결과 저장
    test_embed.obs["pred_celltype"] = preds
    test_embed.write_h5ad(SAVE_DIR / "mode1_test_embed.h5ad")
    np.save(SAVE_DIR / "mode1_predictions.npy", np.array(preds))
    np.save(SAVE_DIR / "mode1_groundtruth.npy", gt)
    sc.pl.umap(test_embed, color=[CELL_TYPE_KEY, "pred_celltype"],
               frameon=False, wspace=0.4,
               save=str(SAVE_DIR / "mode1_umap.png"))
    print(f"결과 저장 완료: {SAVE_DIR}")

    return test_embed, gt  # Mode 2에서 재사용 가능


# ──────────────────────────────────────────────
# Mode 2: CellXGene 아틀라스로 매핑 (faiss 필수)
# ──────────────────────────────────────────────
def run_cellxgene_atlas_mapping(
    index_dir: str,
    test_embed_X: np.ndarray,
    gt: np.ndarray,
):
    if not FAISS_AVAILABLE:
        print("\n[Mode 2 스킵] faiss가 설치되지 않았습니다.")
        print("  설치 방법: conda install -c pytorch faiss-cpu")
        print("  참고: https://github.com/facebookresearch/faiss/wiki/Installing-Faiss")
        return None

    print("\n=== Mode 2: CellXGene 아틀라스 매핑 ===")

    index, meta_labels = load_index(
        index_dir=index_dir,
        use_config_file=False,
        use_gpu=faiss.get_num_gpus() > 0,
    )

    _, idx = index.search(test_embed_X, K_ATLAS)

    voting = []
    for preds in tqdm(meta_labels[idx], desc="Voting"):
        voting.append(vote(preds, return_prob=False)[0])
    voting = np.array(voting)

    print(f"\n정답 레이블 (상위 10개): {gt[:10]}")
    print(f"예측 레이블 (상위 10개): {voting[:10]}")

    # 예시: endothelial 세포 결과 확인
    ids_m = np.where(gt == "endothelial")[0]
    if len(ids_m) > 0:
        print(f"\nEndothelial 세포 {len(ids_m)}개 발견")
        print(f"  예측: {voting[ids_m]}")
        print(f"  정답: {gt[ids_m]}")

    # 결과 저장
    np.save(SAVE_DIR / "mode2_predictions.npy", voting)
    np.save(SAVE_DIR / "mode2_groundtruth.npy", gt)
    print(f"결과 저장 완료: {SAVE_DIR}")

    return voting


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Mode 1 실행 (faiss 없이도 동작)
    test_embed, gt = run_custom_reference_mapping()

    # Mode 2 실행 (faiss 설치 + 인덱스 다운로드 후 index_dir 경로 입력)
    INDEX_DIR = "path_to_faiss_index_folder"  # ← 실제 경로로 변경
    run_cellxgene_atlas_mapping(
        index_dir=INDEX_DIR,
        test_embed_X=test_embed.X,
        gt=gt,
    )
