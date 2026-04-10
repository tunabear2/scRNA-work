# WORKLOG — CellFM 실습

## 2026-04-10 — 초기 기록 작성

- CLAUDE.md, WORKLOG.md 초기 생성
- SETUP.md 작성 — 환경 재현 가이드 (conda 환경, 패키지, 데이터, 체크포인트, 주의사항)

---

## 2026-04-10 — 환경 설정 + 스크립트 변환

### 환경 구성 (scfm conda env)
- `mindspore==2.2.10` (Huawei OBS에서 직접 설치)
- `scanpy==1.10.0`, `scib==1.1.5`, `torch==2.8.0+cu128`
- `gears==0.7.2`, `jupyter`, `ipykernel` (Python scfm 커널 등록)

### 데이터 압축 해제
- `datasets/CellFM.zip` → `datasets/CellFM/` (21개 h5ad + GO_data)

### 스크립트 변환 (notebooks → py)
모든 튜토리얼 노트북을 로컬 실행 가능한 Python 스크립트로 변환.

| 스크립트 | 원본 노트북 | 주요 수정 사항 |
|----------|------------|---------------|
| `scripts/01_cell_annotation_finetune.py` | CellAnnotation_finetune.ipynb | hPancreas 데이터 사용, Celltype→cell_type |
| `scripts/02_cell_annotation_zeroshot.py` | CellAnnotation_zeroshot.ipynb | backbone 동결, 경로 수정 |
| `scripts/03_batch_integration.py` | BatchIntegration.ipynb | PBMC_10K.h5ad, device=0, UMAP 저장 |
| `scripts/04_gene_perturbation.py` | GenePerturbation.ipynb | norman.h5ad→GEARS 포맷, device=0 |
| `scripts/05_binary_gene_function.py` | BinaryclassGeneFunction.ipynb | Gene_classification.h5ad, t1/t2/t3 전체 |
| `scripts/06_multiclass_gene_function.py` | MulticlassGeneFunction.ipynb | GO_data MF/CC/BP, emb 추출 자동화 |
| `scripts/07_lncrna.py` | IdentifyingCelltypelncRNAs.ipynb | PBMC_10K 사용, 경로 버그 수정, CSV 저장 |

### 공통 수정 사항
- 모든 경로 → 절대 경로 (`ROOT_DIR` 기준)
- `device` → `cuda:0` 통일
- `results/{태스크}/` 에 `metrics.json` + 그림 저장
- `scripts/utils/logger.py` — 콘솔+파일 로깅 공통 모듈
- `scripts/run_all.sh` — 전체/개별 실행 셸 스크립트

### 출력 디렉토리 구조
```
results/
  01_cell_annotation_finetune/  → metrics.json, run.log
  02_cell_annotation_zeroshot/  → metrics.json, run.log
  03_batch_integration/         → *.dat, figures/umap_*.png, metrics.json
  04_gene_perturbation/         → gears 결과, metrics.json
  05_binary_gene_function/      → metrics.json, run.log
  06_multiclass_gene_function/  → metrics.json, run.log
  07_lncrna/                    → lncrna_results.csv, metrics.json
checkpoint/
  CellAnnotation/   BatchIntegration/   GeneFunction/   lncRNA/
```
- 프로젝트 구조 정리 및 튜토리얼 현황 파악

---

## 프로젝트 진행 이력 (git 기반 재구성)

### 2024-06-06 — 최초 커밋
- CellFM 공식 구현체 초기 세팅
- 기본 모델 파일 (`model.py`, `attention.py`, `retention.py`, `lora.py` 등) 추가

### 2024-06-15 ~ 06-28 — 초기 업데이트
- 코드베이스 정비 및 튜토리얼 기초 자료 추가

### 2024-08-01 — README 업데이트
- 프로젝트 설명 및 사용법 정리

### 2024-09-17 — 라이선스 추가
- CC BY-NC-ND 4.0 라이선스 파일 추가

### 2024-12-17 — 튜토리얼 확장
- `tutorials/` 폴더 내 다운스트림 태스크 노트북 추가
  - Cell Annotation (finetune / zeroshot)
  - Gene Function Prediction (Binary / Multiclass)
  - Batch Integration
  - lncRNA 동정

### 2025-02-26 — Perturbation 추가
- `tutorials/Perturbation/GenePerturbation.ipynb` 추가
- `tutorials/ChemicalPerturbation/` 폴더 추가
- Gene Perturbation 다운스트림 태스크 실습 자료 완성

### 2025-04-04 — README 업데이트
- 전체 튜토리얼 목록 및 데이터셋 링크 최신화

### 2025-05-26 — 유전자 임베딩 업데이트
- `get_gene_emb.py` 수정 — 유전자 임베딩 추출 로직 개선

### 2025-08-25 ~ 08-26 — README 업데이트
- PyTorch 호환 버전(CellFM-torch) 안내 추가
- Zenodo 데이터셋 링크 업데이트

---

---

## 2026-04-10 — PyTorch 재작성 (scripts 01/02/03/05/07)

### 배경
- MindSpore 2.2.10 GPU 플러그인은 CUDA 11.x 전용이나 H200은 CUDA 12.9 필수
- MindSpore CPU 모드로는 GPU 컴퓨팅 불가 → PyTorch + CellFM-torch 재작성 결정

### CellFM-torch 클론
- `git clone https://github.com/biomed-AI/CellFM-torch.git`
- `layers/utils.py` (SCrna, Prepare, build_dataset), `model.py` (Cell_FM, Finetune_Cell_FM) 활용

### 재작성 내용

| 스크립트 | 방식 | 핵심 변경 |
|----------|------|-----------|
| `01_cell_annotation_finetune.py` | CellFM-torch SCrna + Finetune_Cell_FM | hPancreas obs 컬럼 설정(celltype, feat, batch_id), 전체 파라미터 학습 30 epoch |
| `02_cell_annotation_zeroshot.py` | CellFM-torch SCrna + Finetune_Cell_FM | backbone 동결(cls.* 만 학습), extractor.eval() 유지 |
| `03_batch_integration.py` | 커스텀 PBMCDataset + Cell_FM.net.encode() | 사전학습 모델로 임베딩 추출(학습 없음), scib 메트릭, UMAP |
| `05_binary_gene_function.py` | 순수 PyTorch MLP | gene_emb MindSpore CPU 추출 재사용, acc/F1 평가 |
| `07_lncrna.py` | 커스텀 LncRNADataset + Cell_FM.net.encode() | cls_token ↔ gene_emb 내적으로 attention 대리 점수 계산 |

### 주의사항
- CellFM-torch는 `csv/` 경로를 CWD 기준 상대경로로 읽음 → `os.chdir(ROOT_DIR)` 필수
- `SCrna` 필수 obs: `train`(0/2), `celltype`(str), `feat`(int), `batch_id`(int)
- `celltype_id`는 자동 생성됨

---

---

## 2026-04-10 — 전체 스크립트 실행 완료

### 실행 결과 요약

| # | 태스크 | 결과 | 비고 |
|---|--------|------|------|
| 01 | Cell Annotation Finetune | Best acc **98.01%** (Epoch 12/30) | lr StepLR step_size=5 과격 → 이후 조정 필요 |
| 02 | Cell Annotation Zero-shot | Best acc **47.68%** | Bayesian sampling stochasticity 진동 한계 |
| 03 | Batch Integration | ASW_label/batch=1.0, NMI=0.025 | 배치 효과 미제거, UMAP 시각화 저장 |
| 04 | Gene Perturbation (GEARS) | Pearson **0.975**, Pearson_DE **0.901** | MSE=0.011, MSE_DE=0.226 |
| 05 | Binary Gene Function | t1~t3 avg acc **~80%** | gene_emb MindSpore CPU 추출 후 MLP |
| 06 | Multiclass Gene Function | CC AUPR **0.881**, MF Fmax **0.807** | GO MF/CC/BP top10, 5 epoch |
| 07 | lncRNA 동정 | lncRNA 검출 0개 | PBMC_10K 내 gene_info 매칭 lncRNA 4개뿐 |

### 버그 수정 이력

| 스크립트 | 버그 | 수정 |
|---------|------|------|
| 01, 02 | 평가 시 mask_ratio=0.5 적용 → acc 진동 | `prep_eval(mask_ratio=0.0)` 분리 |
| 03, 07 | `float(self.T[idx])` → `T.copy()` AttributeError | `self.T[idx]` (numpy scalar)로 수정 |
| 03 | scib `metrics()` API 파라미터명 오류 (`ari` → `ari_`) | 언더스코어 접미사 추가 |
| 03 | `louvain` 패키지 누락 | `pip install louvain` |
| 04 | `gears` 패키지가 JS 에셋 컴파일러 | `pip install cell-gears` + `torch_geometric` |
| 04 | `model_initialize()` / `train()` API 버전 불일치 | 불필요 파라미터 제거 |

## 태스크 현황

| 태스크 | 튜토리얼 | 상태 |
|--------|----------|------|
| Cell Annotation (Finetune) | `CellAnnotation/` | ✅ 실행 완료 (98.01%) |
| Cell Annotation (Zero-shot) | `CellAnnotation/` | ✅ 실행 완료 (47.68%) |
| Gene Function (Binary) | `BinaryclassGeneFunction.ipynb` | ✅ 실행 완료 (~80%) |
| Gene Function (Multi) | `MulticlassGeneFunction.ipynb` | ✅ 실행 완료 (CC 0.881) |
| Batch Integration | `BatchIntegration/` | ✅ 실행 완료 (UMAP 저장) |
| Gene Perturbation | `Perturbation/GenePerturbation.ipynb` | ✅ 실행 완료 (15 epoch) |
| lncRNA 동정 | `IdentifyingCelltypelncRNAs.ipynb` | ✅ 실행 완료 (데이터 한계) |
| Chemical Perturbation | `ChemicalPerturbation/` | 미실행 |
