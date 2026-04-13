# CellFM 실습 환경 구축 및 스크립트 검토 작업 로그

작성일: 2026-04-13

---

## 1. 프로젝트 개요

**CellFM** — 1억 개 인간 단일세포 전사체 데이터로 사전학습된 파운데이션 모델.  
원본 구현은 MindSpore 기반이나, 본 실습은 **CellFM-torch** (PyTorch 재구현)를 사용해 **CUDA(GPU) 환경**에서 실행한다.

| 항목 | 내용 |
|------|------|
| GPU | NVIDIA H200 NVL (143 GB) |
| CUDA | 12.9 (드라이버), nvcc 12.9 |
| Python | 3.9 |
| conda 환경 | `cellfm` |
| 사전학습 가중치 | `checkpoint/CellFM_80M_weight.ckpt` (MindSpore 포맷) |
| 유전자 임베딩 | `datasets/CellFM/cellFM_embedding.pt` (PyTorch 포맷) |

---

## 2. 환경 구축

### 2-1. conda 환경 생성

```bash
conda create -n cellfm python=3.9
conda activate cellfm
```

### 2-2. PyTorch (CUDA 12.4 빌드 — CUDA 12.9 호환)

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

설치 버전: `torch==2.6.0+cu124`

### 2-3. 과학/생물정보학 스택

```bash
# numpy 1.x 고정 (일부 패키지 호환성)
pip install numpy==1.26.4 pandas scipy scikit-learn matplotlib seaborn tqdm

# 생물정보학
pip install scanpy anndata leidenalg python-igraph

# scib (Batch Integration 메트릭)
pip install scib
```

### 2-4. MindSpore CPU (체크포인트 읽기 전용)

```bash
pip install mindspore
```

- **사용 목적**: `.ckpt` 파일은 MindSpore protobuf 포맷이라 파일 읽기에만 필요
- **연산은 모두 PyTorch/CUDA** 에서 수행
- 설치 버전: `mindspore==2.8.0` (CPU 전용 빌드)

### 2-5. PyTorch Geometric + GEARS (유전자 퍼터베이션)

```bash
pip install torch-geometric

# torch 2.6.0+cu124 대응 sparse/scatter 바이너리
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# GEARS (유전자 퍼터베이션 모델)
# 주의: PyPI의 'gears'는 웹 에셋 파이프라인으로 전혀 다른 패키지 → cell-gears 설치
pip install cell-gears
```

### 2-6. 설치 완료 패키지 목록 (주요)

| 패키지 | 버전 |
|--------|------|
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torch-geometric | 2.6.1 |
| torch_scatter | 2.1.2+pt26cu124 |
| torch_sparse | 0.6.18+pt26cu124 |
| mindspore | 2.8.0 |
| scanpy | 1.10.3 |
| anndata | 0.10.9 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scipy | 1.13.1 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.9.4 |
| seaborn | 0.13.2 |
| tqdm | 4.67.3 |
| umap-learn | 0.5.12 |
| leidenalg | 0.11.0 |
| h5py | 3.14.0 |
| scib | 1.1.7 |
| cell-gears | 0.1.2 |
| protobuf | 6.33.6 |

---

## 3. 소스 코드 수정 사항

### 3-1. `CellFM-torch/model.py` — MindSpore import lazy 변경

**문제**: `from mindspore.train.serialization import load_checkpoint`가 top-level에 있어
스크립트 실행 시 MindSpore CUDA 미설치 경고가 대량 출력됨.

**수정**: `load_checkpoint` import를 `load_model()` 메서드 내부로 이동 (lazy import).

```python
# 변경 전 (top-level)
from mindspore.train.serialization import load_checkpoint

# 변경 후 (메서드 내부)
def load_model(self, weight, moment):
    if weight:
        from mindspore.train.serialization import load_checkpoint  # ← 여기로 이동
        self.ms_ckpt = load_checkpoint(self.ckpt_path)
        ...
```

또한 deprecated `torch.cuda.amp.GradScaler` → `torch.amp.GradScaler("cuda")` 수정:

```python
# 변경 전
scaler = torch.cuda.amp.GradScaler(init_scale=1.0)
# 변경 후
scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
```

### 3-2. `scripts/run_all.sh` — conda 환경 이름 수정

```bash
# 변경 전
CONDA_ENV="scfm"
# 변경 후
CONDA_ENV="cellfm"
```

---

## 4. scripts/ 폴더 스크립트 검토 결과

### 4-1. 스크립트 ↔ notebook 대응

| 스크립트 | 원본 notebook | 데이터셋 |
|----------|---------------|----------|
| `01_cell_annotation_finetune.py` | `tutorials/CellAnnotation/CellAnnotation_finetune.ipynb` | `hPancreas_train.h5ad`, `hPancreas_test.h5ad` |
| `02_cell_annotation_zeroshot.py` | `tutorials/CellAnnotation/CellAnnotation_zeroshot.ipynb` | 동일 |
| `03_batch_integration.py` | `tutorials/BatchIntegration/BatchIntegration.ipynb` | `PBMC_10K.h5ad` |
| `04_gene_perturbation.py` | `tutorials/Perturbation/GenePerturbation.ipynb` | `datasets/CellFM/norman/` |
| `05_binary_gene_function.py` | `tutorials/BinaryclassGeneFunction.ipynb` | `Gene_classification.h5ad` |
| `06_multiclass_gene_function.py` | `tutorials/MulticlassGeneFunction.ipynb` | `GO_data/{MF,CC,BP}/top10_data/` |
| `07_lncrna.py` | `tutorials/IdentifyingCelltypelncRNAs.ipynb` | `PBMC_10K.h5ad` |

### 4-2. 빠진 실습 내용 (스크립트화 불필요 사유)

| notebook | 사유 |
|----------|------|
| `tutorials/process.ipynb` | 데이터 전처리 유틸리티 모음. 독립 태스크 아님 |
| `CellFM-torch/tutorial/cls_task.ipynb` | script 01과 동일 내용의 CellFM-torch 공식 예제 |
| `tutorials/ChemicalPerturbation/` | Synapse에서 별도 데이터셋(sciplex3) 다운로드 필요. 이미 자체 `train.py` / `evaluate.py` 포함 |

### 4-3. 스크립트에서 발견·수정된 버그

#### Bug 1 — `05_binary_gene_function.py`: fold 값 오류

**문제**: `Gene_classification.h5ad`의 `train_t*` 컬럼 fold 값은 `[-1, 0, 1, 2, 3, 4]` (5-fold, 0-based)인데,
스크립트는 `fold in [1, 2, 3]`으로 3개 fold만 1-indexed로 실행.

```python
# 변경 전
for fold in [1, 2, 3]:

# 변경 후
for fold in [0, 1, 2, 3, 4]:   # 5-fold CV (값 범위: 0~4)
```

그래프 `save_figures`도 5개 fold 기준으로 레이아웃 조정.

#### Bug 2 — `01~03, 07 scripts`: deprecated autocast/GradScaler

PyTorch 2.x에서 `torch.cuda.amp.autocast` / `torch.cuda.amp.GradScaler` deprecated.

```python
# 변경 전
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()
with autocast():

# 변경 후
from torch.amp import GradScaler, autocast
scaler = GradScaler("cuda")
with autocast("cuda"):
```

영향 파일: `01_cell_annotation_finetune.py`, `02_cell_annotation_zeroshot.py`,
`03_batch_integration.py`, `07_lncrna.py`, `CellFM-torch/model.py`

### 4-4. 설계 관련 참고사항 (버그 아님)

| 항목 | notebook 원본 | 스크립트 |
|------|--------------|----------|
| Script 03 Batch Integration | 30 epoch 파인튜닝 후 임베딩 추출 (MindSpore 전용 `finetune_intergration_model`) | 사전학습 가중치로 직접 임베딩 추출 (CellFM-torch에 해당 통합 모델 없음) |
| Script 04 Gene Perturbation | GEARS custom fork (`model_type='emb'`) | 표준 `cell-gears 0.1.2` API (`model_initialize(hidden_size=...)`) |
| Script 07 lncRNA attention | MindSpore attention weight 직접 추출 | cls_token ↔ gene embedding dot-product 유사도로 대체 |

---

## 5. 체크포인트 로딩 흐름

```
checkpoint/CellFM_80M_weight.ckpt  (MindSpore protobuf 포맷)
         │
         │  mindspore.train.serialization.load_checkpoint()  [CPU 전용, 연산 없음]
         ▼
MindSpore Parameter dict  →  .asnumpy()  →  numpy array
         │
         │  torch.tensor(numpy_array)
         ▼
PyTorch state_dict  →  net.load_state_dict()
         │
         │  .to("cuda:0")
         ▼
H200 CUDA  (이후 모든 학습/추론)
```

---

## 6. 실행 방법

### 전체 실행

```bash
cd ~/DW/CellFM
bash scripts/run_all.sh
```

### 개별 태스크 실행

```bash
conda activate cellfm
cd ~/DW/CellFM

python scripts/01_cell_annotation_finetune.py   # Cell Annotation (전체 파인튜닝)
python scripts/02_cell_annotation_zeroshot.py   # Cell Annotation (백본 동결)
python scripts/03_batch_integration.py          # Batch Effect Integration
python scripts/04_gene_perturbation.py          # Gene Perturbation (GEARS)
python scripts/05_binary_gene_function.py       # Binary Gene Function (t1/t2/t3, 5-fold)
python scripts/06_multiclass_gene_function.py   # Multiclass GO Function (MF/CC/BP)
python scripts/07_lncrna.py                    # lncRNA 세포 유형별 동정
```

### 결과 위치

```
results/
├── 01_cell_annotation_finetune/
│   ├── metrics.json
│   ├── predicted_labels.csv
│   ├── run.log
│   └── figures/
│       ├── learning_curve.png
│       ├── confusion_matrix.png
│       ├── umap_true_label.png
│       └── umap_pred_label.png
├── 02_cell_annotation_zeroshot/      (동일 구조)
├── 03_batch_integration/
│   ├── metrics.json, run.log
│   ├── PBMC_embeddings.npy
│   ├── PBMC_integrated.h5ad
│   └── figures/ (umap_cell_type.png, umap_batch.png)
├── 04_gene_perturbation/
│   ├── metrics.json, run.log
│   ├── predicted_expression.csv
│   └── figures/ (scatter_pred_vs_truth.png, top_perturbations_pearson.png)
├── 05_binary_gene_function/
│   ├── metrics.json, run.log
│   └── figures/ (acc_f1_bar.png, gene_embedding_umap.png)
├── 06_multiclass_gene_function/
│   ├── metrics.json, run.log
│   └── figures/ (aupr_fmax_bar.png, pr_curves.png)
└── 07_lncrna/
    ├── metrics.json, run.log
    ├── lncrna_results.csv
    └── figures/ (umap_cell_type.png, umap_top_lncrna_score.png, lncrna_attn_bar_*.png)

checkpoint/
├── CellAnnotation/
│   ├── hPancreas_finetune_best.pth
│   └── hPancreas_zeroshot_best.pth
└── GeneFunction/
    └── {MF,CC,BP}_top10_best.pth
```

---

## 7. 트러블슈팅

### MindSpore CUDA 경고 (정상)

```
[ERROR] Cuda version is not found ...
[WARNING] Can not found cuda libs ...
```

MindSpore는 `.ckpt` 파일 읽기(CPU)에만 사용하므로 CUDA 미검출 경고는 무시해도 됨.

### GEARS 패키지 혼동 주의

```bash
pip install gears       # ← 웹 에셋 파이프라인 패키지 (잘못된 것)
pip install cell-gears  # ← 유전자 퍼터베이션 GEARS (올바른 것)
```

### GPU 메모리 부족 시

`BATCH` / `BATCH_SIZE` 상수를 각 스크립트 상단에서 줄여서 실행:

```python
BATCH = 8   # 기본값 16에서 줄이기
```
