# CellFM 실습 환경 셋업 가이드

다른 환경에서 이 실습을 재현하기 위한 단계별 가이드.

## 시스템 요구사항

| 항목 | 요구 사항 |
|------|-----------|
| GPU | NVIDIA (CUDA 12.x 지원) |
| CUDA | 12.8+ (H200 기준) |
| OS | Linux (Ubuntu 20.04+) |
| conda | miniconda 또는 anaconda |
| 디스크 | 약 30GB 이상 (데이터 + 체크포인트) |

> **주의**: MindSpore 2.2.10 GPU 플러그인은 CUDA 11.x 전용이므로 H200(CUDA 12.x) 환경에서는  
> MindSpore를 CPU 모드로만 사용하고, 실제 GPU 연산은 PyTorch로 처리한다.

---

## 1. 저장소 클론

```bash
# CellFM 메인 (실습 스크립트 포함)
git clone https://github.com/biomed-AI/CellFM.git
cd CellFM

# CellFM-torch (PyTorch 호환 버전 — scripts/ 에서 import 필요)
git clone https://github.com/biomed-AI/CellFM-torch.git
```

디렉토리 구조:
```
CellFM/               ← 이 저장소
├── CellFM-torch/     ← 위에서 클론한 것을 여기에 위치
├── scripts/
├── tutorials/
└── ...
```

---

## 2. conda 환경 생성

```bash
conda create -n scfm python=3.9 -y
conda activate scfm
```

---

## 3. 패키지 설치

### 3-1. MindSpore (CPU 모드 — CUDA 12.x 환경)

```bash
pip install mindspore==2.2.10
```

> CUDA 11.x 환경이라면 GPU 버전 설치 가능:
> `pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.10/MindSpore/unified/x86_64/mindspore-2.2.10-cp39-cp39-linux_x86_64.whl`

### 3-2. PyTorch (CUDA 12.8 기준)

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

CUDA 버전이 다른 경우 [PyTorch 공식 사이트](https://pytorch.org/get-started/locally/)에서 맞는 버전 선택.

### 3-3. 단일세포 분석 라이브러리

```bash
pip install scanpy==1.10.0 scib==1.1.5 anndata==0.10.9
```

### 3-4. 그래프 기반 라이브러리 (GEARS 의존성)

```bash
pip install torch-geometric==2.6.1
# torch_scatter/sparse는 torch-geometric이 자동 처리 (버전에 따라 별도 설치 필요할 수 있음)
```

### 3-5. GEARS (Gene Perturbation)

```bash
pip install cell-gears==0.1.2
```

> `gears` (PyPI)가 아닌 `cell-gears`가 올바른 패키지.

### 3-6. 클러스터링 라이브러리

```bash
pip install louvain==0.8.2 leidenalg==0.11.0
```

### 3-7. 기타 필수 라이브러리

```bash
pip install numpy==1.26.4 scipy==1.13.1 pandas scikit-learn matplotlib seaborn umap-learn
```

### 3-8. Jupyter (선택)

```bash
pip install jupyter jupyterlab ipykernel
python -m ipykernel install --user --name scfm --display-name "Python (scfm)"
```

---

## 4. 데이터 다운로드

Zenodo에서 데이터셋 다운로드: https://zenodo.org/records/15138665

```bash
mkdir -p datasets/CellFM
# Zenodo에서 CellFM.zip 다운로드 후:
unzip CellFM.zip -d datasets/CellFM/
```

다운로드 후 `datasets/CellFM/` 내에 있어야 할 주요 파일:

```
datasets/CellFM/
├── hPancreas_train.h5ad      # Cell Annotation 학습
├── hPancreas_test.h5ad       # Cell Annotation 평가
├── PBMC_10K.h5ad             # Batch Integration, lncRNA
├── norman.h5ad               # Gene Perturbation
├── Gene_classification.h5ad  # Binary Gene Function (t1/t2/t3 포함)
├── GO_data/                  # Multiclass Gene Function (MF/CC/BP)
├── norman/                   # GEARS 전처리 데이터
├── essential_all_data_pert_genes.pkl
└── gene2go_all.pkl
```

---

## 5. 체크포인트 다운로드

Huggingface에서 사전학습 가중치 다운로드: https://huggingface.co/ShangguanNingyuan/CellFM

```bash
mkdir -p checkpoint
# 다운로드 후 배치:
# checkpoint/base_weight.ckpt        ← MindSpore 형식 (gene_emb 추출용)
# checkpoint/CellFM_80M_weight.ckpt  ← PyTorch 형식 (scripts 01~03, 07)
```

또는 CellFM-torch 안에 넣어도 됨 (스크립트 경로 참고):

```
checkpoint/
├── base_weight.ckpt          # MindSpore 사전학습 가중치
├── CellFM_80M_weight.ckpt    # PyTorch 호환 가중치 (CellFM-torch)
├── CellAnnotation/           # 01, 02 파인튜닝 저장 위치
├── BatchIntegration/         # 03 저장 위치
├── GeneFunction/             # 05, 06 저장 위치
└── lncRNA/                   # 07 저장 위치
```

---

## 6. 환경 검증

```bash
conda activate scfm
python - <<'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

import mindspore as ms
print(f"MindSpore: {ms.__version__}")

import scanpy; print(f"scanpy: {scanpy.__version__}")
import scib;   print(f"scib: {scib.__version__}")
import gears;  print(f"gears: OK")
import louvain; print(f"louvain: OK")
EOF
```

정상 출력 예시:
```
PyTorch: 2.8.0
CUDA available: True
CUDA version: 12.8
MindSpore: 2.2.10
scanpy: 1.10.0
scib: 1.1.5
gears: OK
louvain: OK
```

---

## 7. 스크립트 실행

```bash
cd /path/to/CellFM

# 전체 태스크 순서대로 실행
bash scripts/run_all.sh

# 특정 태스크만 실행 (번호 지정)
bash scripts/run_all.sh 1   # Cell Annotation Finetune
bash scripts/run_all.sh 2   # Cell Annotation Zero-shot
bash scripts/run_all.sh 3   # Batch Integration
bash scripts/run_all.sh 4   # Gene Perturbation
bash scripts/run_all.sh 5   # Binary Gene Function
bash scripts/run_all.sh 6   # Multiclass Gene Function
bash scripts/run_all.sh 7   # lncRNA 동정
```

결과는 `results/{태스크번호}/` 에 저장됨:
- `metrics.json` — 정량 평가 지표
- `run.log` — 실행 로그
- `figures/` — UMAP 등 시각화 (해당 태스크)

---

## 8. 알려진 이슈 및 주의사항

### CellFM-torch 경로 문제
`CellFM-torch`는 `csv/` 디렉토리를 **현재 작업 디렉토리** 기준 상대 경로로 읽는다.  
스크립트 상단에 반드시 `os.chdir(ROOT_DIR)` 호출 필요.

### SCrna obs 필수 컬럼
`CellFM-torch`의 `SCrna` 클래스가 요구하는 obs 컬럼:

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `train` | int (0 또는 2) | 0=학습, 2=테스트 |
| `celltype` | str | 세포 유형 레이블 |
| `feat` | int | 발현 유전자 수 (n_genes_by_counts 등) |
| `batch_id` | int | 배치 ID (단일 배치면 0으로 채움) |

`celltype_id`는 SCrna 내부에서 자동 생성됨.

### hPancreas obs 컬럼명
원본 h5ad 파일의 컬럼명이 `Celltype`(대문자)이므로 로드 후 rename 필요:
```python
adata.obs.rename(columns={'Celltype': 'celltype'}, inplace=True)
```

### 평가 시 mask_ratio
평가 데이터 준비 시 `mask_ratio=0.0`으로 고정해야 정확한 acc 측정 가능.  
학습용 `prep_data(mask_ratio=0.5)`와 구분해서 사용할 것.

### scib metrics API
`scib.metrics.metrics()` 호출 시 파라미터명에 언더스코어 접미사 필요:
```python
# 올바른 사용
results = scib.metrics.metrics(..., ari_=True, nmi_=True, ...)
```

### Gene Perturbation (GEARS)
```bash
# PyPI의 gears가 아닌 cell-gears 설치
pip install cell-gears

# norman 데이터는 전처리된 형태로 datasets/CellFM/norman/ 에 있어야 함
# (norman.h5ad만으로는 부족, GEARS 포맷 전처리 필요)
```

---

## 9. 실행 결과 요약 (참고)

| # | 태스크 | 주요 지표 | 비고 |
|---|--------|-----------|------|
| 01 | Cell Annotation (Finetune) | Acc **98.01%** | hPancreas, 30 epoch |
| 02 | Cell Annotation (Zero-shot) | Acc **47.68%** | backbone 동결 |
| 03 | Batch Integration | ASW=1.0, NMI=0.025 | PBMC_10K, UMAP 저장 |
| 04 | Gene Perturbation | Pearson **0.975** | GEARS, norman 데이터 |
| 05 | Binary Gene Function | Acc **~80%** | t1/t2/t3, MindSpore CPU emb |
| 06 | Multiclass Gene Function | CC AUPR **0.881** | GO MF/CC/BP |
| 07 | lncRNA 동정 | — | PBMC_10K (lncRNA 4개 검출) |
