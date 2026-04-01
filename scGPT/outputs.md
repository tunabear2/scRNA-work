# 스크립트별 출력 결과물

## annotation.py — 세포 타입 주석
**출력 경로:** `./results/annotation/dev_{데이터셋}-{타임스탬프}/`

| 파일 | 내용 |
|------|------|
| `model.pt` | 최적 모델 가중치 |
| `annotated_output.h5ad` | 예측 결과 포함된 AnnData |
| `results.pkl` | 평가 결과 딕셔너리 |
| `confusion_matrix.png` | 혼동 행렬 시각화 |
| `results.png` | UMAP 등 시각화 |
| `vocab.json` | 사용된 유전자 사전 |
| `run.log` | 학습 로그 |

---

## integration.py — 배치 통합
**출력 경로:** `./results/integration/dev_{데이터셋}-{타임스탬프}/`

| 파일 | 내용 |
|------|------|
| `best_model.pt` | 최적 모델 |
| `model_e{epoch}.pt` | 에포크별 체크포인트 |
| `embeddings_batch_umap[cls]_e{epoch}.png` | 배치별 색상 UMAP |
| `embeddings_celltype_umap[cls]_e{epoch}.png` | 세포 타입별 UMAP |
| `run.log` | 학습 로그 |

---

## perturbation.py — 섭동 예측
**출력 경로:** `./results/perturbation/dev_perturb_{데이터셋}-{타임스탬프}/`

| 파일 | 내용 |
|------|------|
| `best_model.pt` | 최적 모델 |
| `{섭동조건}.png` | 조건별 예측 vs 실제 비교 플롯 |
| `test_metrics.json` | 피어슨 상관계수 등 평가 지표 |
| `run.log` | 학습 로그 |

---

## GRN_inference.py — 유전자 조절 네트워크 추론
**출력 경로:** `./results/GRN_inference/`

| 파일 | 내용 |
|------|------|
| `gene_program_activation.png` | 세포 타입별 유전자 프로그램 히트맵 |
| `gene_network_program_{id}.png` | 유전자 상호작용 네트워크 시각화 |
| `pathway_enrichment.csv` | Reactome 경로 농축 분석 결과 |

---

## attentionGRN.py — 어텐션 기반 GRN
**출력 경로:** `./results/attentionGRN/`

| 파일 | 내용 |
|------|------|
| `clustermap_{TF이름}.png` | 어텐션 차이 클러스터 히트맵 |
| `gene_network_{TF이름}.png` | TF→타겟 유전자 방향성 네트워크 |
| `pathway_enrichment_{TF이름}.csv` | TF별 경로 농축 분석 결과 |

---

## multiomics_finetune.py — 멀티오믹스 통합 (RNA + Protein)
**출력 경로:** `./results/multiomics_finetune/dev_{데이터셋}-{타임스탬프}/`

| 파일 | 내용 |
|------|------|
| `best_model.pt` | 최적 모델 |
| `model_e{epoch}.pt` | 에포크별 체크포인트 |
| `*_umap*.png` | 배치/세포타입별 UMAP |
| `run.log` | 학습 로그 |

---

## reference_mapping.py — 레퍼런스 매핑
**출력 경로:** `./results/reference_mapping/`

| 파일 | 내용 |
|------|------|
| `mode1_test_embed.h5ad` | 예측 세포 타입 포함 임베딩 |
| `mode1_predictions.npy` | 예측 레이블 |
| `mode1_groundtruth.npy` | 실제 레이블 |
| `mode1_umap.png` | 예측 vs 실제 UMAP |

---

## 공통 패턴

- **Fine-tuning 스크립트** (`annotation`, `integration`, `perturbation`, `multiomics`): 타임스탬프가 붙은 디렉토리에 모델 체크포인트 + UMAP 시각화 저장
- **분석 스크립트** (`GRN_inference`, `attentionGRN`, `reference_mapping`): 고정 경로에 플롯/CSV 저장
