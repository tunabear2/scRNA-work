# scFoundation 작업 로그

작성일: 2026-04-03

---

## 1. GitHub 푸쉬

### 작업 내용
- `biomap-research/scFoundation` 원본 레포를 로컬에 clone한 상태에서,  
  `tunabear2/scRNA-work` 레포의 `scFoundation/` 서브폴더로 코드를 푸쉬함.
- `scBERT/`, `scGPT/` 등 기존 폴더는 그대로 유지됨.

### 제외한 파일
- `*.zip` 대용량 데이터 파일
- `m9.figshare.*` 파일

### Remote 정보
| Remote | URL |
|--------|-----|
| origin | https://github.com/biomap-research/scFoundation |
| scrna  | https://github.com/tunabear2/scRNA-work.git |

---

## 2. 데이터 압축 해제

`scFoundation/` 폴더 내 모든 `.zip` 파일을 각 폴더에 압축 해제함.

| zip 파일 | 압축 해제 위치 |
|----------|---------------|
| `annotation/cell_type_rawdata.zip` | `annotation/cell_type_rawdata/` |
| `annotation/annotation_data.zip` | `annotation/data/` |
| `mapping/data_mapping.zip` | `mapping/data_mapping/` |
| `model/model_example.zip` | `model/examples/` |
| `SCAD/data_SCAD_split_norm.zip` | `SCAD/data/` |
| `DeepCDR/data/GDSC/drug_graph_feat.zip` | `DeepCDR/data/GDSC/` |
| `GEARS/gse90546_*.h5ad.zip` | `GEARS/` |
| `GEARS/gse133344_*.h5ad.zip` | `GEARS/` |
| `GEARS/gse90063_*.h5ad.zip` | `GEARS/` |
| `genemodule/data_genemodule.zip` | `genemodule/data/` |
| `ablation/data_ablation.zip` | `ablation/data/` |

### 주요 대용량 파일
| 파일 | 크기 |
|------|------|
| `genemodule/data/cd8t_b_mono_geneemb_...npy` | 12 GB |
| `genemodule/data_genemodule.zip` | 11 GB |
| `mapping/data_mapping/merged.h5ad` | 3.5 GB |
| `mapping/data_mapping/scfadata.h5ad` | 3.5 GB |
| `annotation/cell_type_rawdata/zheng/data_train_count.npy` | 7.6 GB |

---

## 3. conda 환경 구성 (`scfoundation`)

모든 실습은 `scfoundation` conda 환경에서 진행.

```bash
conda activate scfoundation
```

### 설치된 패키지
| 패키지 | 용도 |
|--------|------|
| numpy, pandas, scipy | 데이터 처리 |
| matplotlib, seaborn | 시각화 |
| scikit-learn, tqdm | ML 유틸 |
| scanpy, anndata | 단일세포 분석 |
| einops, local-attention | 모델 구조 |
| torch 2.5.1 + CUDA 12.1 | 딥러닝 |
| pyscenic 0.12.1 | Gene Module Inference |
| loompy, networkx, adjustText | Gene Module 시각화 |
| scib | Cell Mapping 평가 |
| jupyter, nbconvert | 노트북 실행 |

---

## 4. 실습 준비 (3가지 다운스트림 태스크)

### 4-1. Gene Module Inference
- **목적**: 유전자 임베딩을 기반으로 유전자 모듈 및 규제 네트워크 추론
- **실행 파일**: `genemodule/plot_geneemb.py`
- **주요 입력 데이터**:
  - `genemodule/data/cd8t_b_mono_geneemb_01B-resolution_singlecell_gene_embedding_f1_resolution.npy` (12 GB)
  - `genemodule/data/zheng68k_train_var.csv`
  - `genemodule/data/allTFs_hg38.txt`
  - `genemodule/data/zheng_downsampled_cd8t_b_mono.h5ad`
  - `genemodule/auc_mtx_1000.csv`
- **주요 출력**:
  - gene UMAP 클러스터링 시각화
  - T cell gene module 네트워크 그래프 (`figures/T_genemodule.pdf`)
  - GRN (Gene Regulatory Network) 파일 (`scf_grn_1000.tsv`)
  - pySCENIC RSS 결과 (`RSS.csv`, `RSS.pdf`)
- **특이사항**: pySCENIC ctx/aucell 단계는 Docker 명령으로 실행 (노트북 내 주석 처리됨)

### 4-2. Cell Mapping
- **목적**: Organoid 데이터를 in vivo 데이터에 매핑하고 BBKNN으로 통합
- **실행 파일**: `mapping/mapping-publish.py`
- **주요 입력 데이터**:
  - `mapping/data_mapping/merged.h5ad` (3.5 GB)
  - `mapping/data_mapping/scfadata.h5ad` (3.5 GB)
  - `mapping/data_mapping/rawmerged.h5ad`
  - `mapping/data_mapping/organoid_scbert_embedding.npy`
- **주요 출력**:
  - UMAP 시각화 (raw / scFoundation embedding 비교)
  - scib 지표 (cLISI, iLISI)
- **특이사항**: 이미 생성된 embedding 결과물을 불러와 시각화하는 코드

### 4-3. Cell Type Annotation
- **목적**: scFoundation 임베딩 기반 세포 유형 분류 및 CellTypist 비교
- **실행 파일**: `annotation/celltype-plot.py`
- **주요 입력 데이터**:
  - `annotation/data/seg-emb.pkl`, `seg-cellemb.pkl`
  - `annotation/data/zheng-emb-2mlp.pkl`, `zheng-cellemb-2mlp.pkl`
  - `annotation/data/Segerstolpe-test-label.npy`, `zheng-test-label.npy`
  - `annotation/data/celltypist_0806_seg.h5ad`, `celltypist_0806_zheng68k.h5ad`
- **주요 출력**:
  - Segerstolpe / Zheng68K 데이터셋 UMAP (true vs predicted label)
  - Classification report (sklearn)
  - CellTypist 결과 비교 시각화

---

## 5. 로컬 실행 방법

각 태스크는 Jupyter Notebook을 Python 스크립트로 변환한 파일로 실행.

```bash
conda activate scfoundation
cd ~/DW/scFoundation

# Gene Module Inference
python genemodule/plot_geneemb.py

# Cell Mapping
python mapping/mapping-publish.py

# Cell Type Annotation
python annotation/celltype-plot.py
```

> 시각화 결과는 각 폴더의 `figures/` 디렉터리에 저장됨 (matplotlib Agg 백엔드 사용).

---

## 6. 참고사항

- 원본 논문: [scFoundation - Nature Methods 2024](https://www.nature.com/articles/s41592-024-02305-7)
- 데이터 출처: [Figshare](https://dx.doi.org/10.6084/m9.figshare.24049200)
- API (구버전 종료): 2024년 4월 30일 종료, 신규 플랫폼 https://aigp.biomap.com/ 으로 이전
