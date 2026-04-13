# CellFM 실습 프로젝트

## 프로젝트 개요

**CellFM** — 1억 개 인간 단일세포 전사체 데이터로 사전학습된 대규모 단일세포 파운데이션 모델.

- 공식 구현체: [biomed-AI/CellFM](https://github.com/biomed-AI/CellFM)
- Huggingface 체크포인트: [ShangguanNingyuan/CellFM](https://huggingface.co/ShangguanNingyuan/CellFM)
- 데이터셋: [Zenodo](https://zenodo.org/records/15138665)

## 프레임워크

- **PyTorch 호환 버전**: [CellFM-torch](https://github.com/biomed-AI/CellFM-torch) (원본 가중치 호환)

## 환경 설정

```shell
conda create -n cellfm python=3.9
conda activate cellfm
```

## 주요 파일 구조

| 파일/폴더 | 역할 |
|-----------|------|
| `model.py` | CellFM 모델 아키텍처 |
| `attention.py` | 어텐션 메커니즘 |
| `retention.py` | Retention 모듈 |
| `lora.py` | LoRA 파인튜닝 모듈 |
| `train.py` | 학습 스크립트 |
| `data_process.py` | 데이터 전처리 |
| `get_gene_emb.py` | 유전자 임베딩 추출 |
| `config.py` | 하이퍼파라미터 설정 |
| `loss_function.py` | 손실 함수 |
| `metrics.py` | 평가 지표 |
| `earlystop.py` | 조기 종료 |
| `utils.py` | 유틸리티 함수 |
| `tutorials/` | 다운스트림 태스크 튜토리얼 |
| `csv/` | 유전자 정보 CSV 파일 |

## 다운스트림 태스크 (Tutorials)

1. **Cell Annotation** — `tutorials/CellAnnotation/`
   - `CellAnnotation_finetune.ipynb` : 파인튜닝 방식
   - `CellAnnotation_zeroshot.ipynb` : 제로샷 방식
2. **Gene Function Prediction** — `tutorials/BinaryclassGeneFunction.ipynb`, `MulticlassGeneFunction.ipynb`
3. **Batch Effect Correction** — `tutorials/BatchIntegration/BatchIntegration.ipynb`
4. **Gene Perturbation** — `tutorials/Perturbation/GenePerturbation.ipynb`
5. **lncRNA 동정** — `tutorials/IdentifyingCelltypelncRNAs.ipynb`
