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

## 6. Fine-tuning 분석

### 제공 파일 구성

| 파일 | 내용 |
|------|------|
| `model/finetune_model.py` | Fine-tuning 가이드용 템플릿 코드 |
| `model/load.py` | 모델 로딩 유틸리티 |
| `GEARS/gears/gears.py` | 실제 end-to-end fine-tuning 코드 (GEARS 전용) |

---

### Pretrain 코드는 미제공

`model/training_hyperparameter.txt`에 pretrain 설정이 기록되어 있으나, 실제 학습 코드는 공개하지 않음.

```
trainer.params.num_nodes=64   # GPU 노드 64개
trainer.params.precision=bf16
training.batch_size=128
optimizer.params.lr=1e-4
```

연구진이 내부적으로 DeepSpeed + 64 GPU 노드로 학습한 결과물(`models.ckpt`)만 배포.

---

### 다운스트림 태스크별 Fine-tuning 방식

| 태스크 | 방식 | 학습 코드 제공 여부 |
|--------|------|-------------------|
| Cell Mapping | scFoundation frozen → 임베딩 추출 → 시각화 | ❌ |
| Gene Module | scFoundation frozen → 임베딩 추출 → 클러스터링 | ❌ |
| Cell Type Annotation | scFoundation frozen → MLP 분류기 학습 → 결과 저장 | ❌ (결과물만 제공) |
| GEARS | scFoundation + GEARS 통합 → end-to-end fine-tuning | ✅ |

---

### Cell Type Annotation 실제 구조

`seg-emb.pkl`, `zheng-emb-2mlp.pkl` 파일의 shape 분석 결과:
- `seg-emb`: `(854, 13)` → 854개 세포, 13개 클래스의 **MLP 분류 logit**
- `zheng-emb-2mlp`: `(6595, 11)` → 6595개 세포, 11개 클래스의 **MLP 분류 logit**

즉, 연구진이 내부적으로 2-layer MLP 분류기를 fine-tuning한 후 결과만 저장해서 제공.  
**학습 코드는 미제공** → 다른 데이터로 annotation fine-tuning 하려면 직접 구현 필요.

---

### scFoundation vs 다른 scFM 비교

| | scFoundation | scGPT / scBERT |
|--|--|--|
| Pretrained 모델 제공 | ✅ | ✅ |
| Annotation fine-tuning 코드 | ❌ | ✅ |
| 제공 형태 | 학습 완료된 결과물(.pkl, .npy) | 학습 코드 전체 |

---

### GEARS Fine-tuning 옵션

GEARS는 `finetune_method` 파라미터로 fine-tuning 방식을 선택 가능:

| 옵션 | 동작 |
|------|------|
| `'frozen'` | scFoundation 가중치 완전 고정, GEARS 레이어만 학습 |
| `'finetune_lr_1'` | scFoundation은 lr×0.1로 천천히 학습, GEARS는 lr로 학습 |
| `None` | 전체 모델 모두 학습 |

---

### 외부 데이터로 scFoundation 사용하는 방법

Fine-tuning 코드가 없더라도 외부 `.h5ad` 데이터에 scFoundation 적용 가능.  
경로 수정만으로 기존 실습 코드 재사용 가능.

```python
# Step 1. 유전자 이름을 19264개 목록에 맞춤 (없는 유전자는 0으로 padding)
gene_list_df = pd.read_csv('OS_scRNA_gene_index.19264.tsv', delimiter='\t')
gene_list = list(gene_list_df['gene_name'])
X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)

# Step 2. 임베딩 추출
# python model/get_embedding.py --input_type singlecell --output_type cell \
#   --data_path ./my_data.h5ad --save_path ./output/ --pre_normalized F

# Step 3. 추출된 임베딩을 실습 코드에 연결
scfemb = np.load('./output/my_data_embedding.npy')
```

---

## 7. 참고사항

- 원본 논문: [scFoundation - Nature Methods 2024](https://www.nature.com/articles/s41592-024-02305-7)
- 데이터 출처: [Figshare](https://dx.doi.org/10.6084/m9.figshare.24049200)
- API (구버전 종료): 2024년 4월 30일 종료, 신규 플랫폼 https://aigp.biomap.com/ 으로 이전
