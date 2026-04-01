# scGPT Input Data Reference

각 스크립트의 하드코딩된 입력 파일 경로와 다른 데이터로 실습할 때 필요한 파일 형식을 정리합니다.

---

## 공통 구조: 모델 디렉토리 (3개 파일 필수)

모든 스크립트가 동일한 구조를 요구합니다:

```
data/models/<model_name>/
├── best_model.pt   # PyTorch 체크포인트
├── args.json       # 모델 설정 (하이퍼파라미터)
└── vocab.json      # Gene vocabulary (16,906 genes)
```

---

## 스크립트별 입력 파일 정리

### 1. `annotation.py`

| 변수 | 하드코딩 경로 | 형식 | 필수 컬럼/구조 |
|------|-------------|------|--------------|
| `config.load_model` | `./data/models/pretrain_human` | 모델 디렉토리 | 위 3파일 |
| `cli_args.train_data` | `./data/annotation/c_data.h5ad` | AnnData H5AD | `adata.obs["celltype"]` (label), raw count matrix |
| `cli_args.test_data` | `./data/annotation/filtered_ms_adata.h5ad` | AnnData H5AD | 동일, `celltype` 컬럼 |

> CLI로 오버라이드 가능: `--train_data`, `--test_data`

---

### 2. `integration.py`

| 변수 | 하드코딩 경로 | 형식 |
|------|-------------|------|
| `config.load_model` | `./data/models/pretrain_human` | 모델 디렉토리 |
| 데이터 | 없음 — `scvi.data.pbmc_dataset()` API로 자동 다운로드 | scvi-tools 내장 |

---

### 3. `perturbation.py`

| 변수 | 하드코딩 경로 | 형식 | 비고 |
|------|-------------|------|------|
| `load_model` | `./data/models/pretrain_human` | 모델 디렉토리 | |
| `PertData(...)` | `./data/perturbation` | GEARS PertData | `data_name = "adamson"` or `"norman"`, GEARS 패키지가 자동 다운로드 |

---

### 4. `GRN_inference.py`

| 변수 | 하드코딩 경로 | 형식 | 필수 구조 |
|------|-------------|------|----------|
| `MODEL_DIR` | `./data/models/pretrain_bc` | 모델 디렉토리 | 혈액 특화 모델 (`pretrain_human`과 다름!) |
| `DATA_PATH` | `./data/GRN_inference/Immune_ALL_human.h5ad` | AnnData H5AD | 면역 세포 데이터, raw counts |

---

### 5. `attentionGRN.py`

| 변수 | 하드코딩 경로 | 형식 | 비고 |
|------|-------------|------|------|
| `model_dir` | `./data/attentionGRN/finetuned_scGPT_adamson` | 모델 디렉토리 | pretrain_human이 아닌 fine-tuned 모델 |
| `data_dir` | `./data/perturbation/adamson/perturb_processed.h5ad` | AnnData H5AD | GEARS로 처리된 Adamson 데이터 |
| `TF_name` | `./data/attentionGRN/reference/BHLHE40.10.tsv` | TSV | CHIP-Atlas TF 타겟 목록 |

---

### 6. `multiomics_finetune.py`

| 변수 | 하드코딩 경로 | 형식 | 필수 구조 |
|------|-------------|------|----------|
| `config.load_model` | `./data/models/pretrain_human` | 모델 디렉토리 | |
| `sc.read(...)` | `./data/multiomics_finetune/BMMC_processed.h5ad` | AnnData H5AD | CITE-seq (RNA + Protein), `adata.obsm["protein_expression"]` 필요 |

---

### 7. `reference_mapping.py`

| 변수 | 하드코딩 경로 | 형식 | 필수 구조 |
|------|-------------|------|----------|
| `MODEL_DIR` | `./data/models/pretrain_human` | 모델 디렉토리 | |
| train data | `./data/reference_mapping/demo_train.h5ad` | AnnData H5AD | `adata.obs["celltype"]` (label), raw counts |
| test data | `./data/reference_mapping/demo_test.h5ad` | AnnData H5AD | 동일 구조 |
| `INDEX_DIR` | `"path_to_faiss_index_folder"` (플레이스홀더) | FAISS index | Mode 2에서만 필요 |

---

## 다른 데이터로 실습할 때 H5AD 파일 최소 요구사항

```python
import anndata as ad

adata.X          # raw count matrix (cell × gene), int 또는 float
adata.var_names  # gene symbol 목록 (예: "CD3D", "MS4A1", ...)
                 # scGPT vocab과 겹치는 유전자가 많을수록 좋음
adata.obs["celltype"]    # annotation.py, reference_mapping.py 필수
adata.obs["batch_id"]    # integration.py, multiomics_finetune.py 필요
```

**핵심 제약:**
- 유전자 이름이 `vocab.json` (16,906 human genes)과 매칭되어야 함 → **human gene symbol** 사용 필수
- `multiomics_finetune.py`만 CITE-seq (RNA + Protein 동시 측정) 데이터 필요, 나머지는 단일 RNA-seq 가능
- `GRN_inference.py`, `attentionGRN.py`는 `pretrain_bc` 모델 필요 (일반 `pretrain_human`과 별도 다운로드)

---

## 모델 다운로드

| 모델 이름 | 사용 스크립트 | 설명 |
|----------|------------|------|
| `pretrain_human` | annotation, integration, perturbation, multiomics_finetune, reference_mapping | 인간 전신 pre-trained 모델 |
| `pretrain_bc` | GRN_inference | 혈액/면역 세포 특화 모델 |
| `finetuned_scGPT_adamson` | attentionGRN | Adamson perturbation 데이터로 fine-tuned |

다운로드 링크는 `README.md`의 **Pretrained Model Zoo** 섹션 참조.
