# scGPT Tutorials - Python Scripts (Linux Local)

Jupyter Notebook에서 변환된 로컬 Linux 실행용 Python 스크립트입니다.

## 파일 목록

| 파일 | 내용 | 특이사항 |
|------|------|---------|
| `Tutorial_Annotation__2_.py` | 세포 유형 어노테이션 fine-tuning | wandb shim 적용 |
| `Tutorial_Attention_GRN__1_.py` | Attention 기반 GRN 추론 | flash-attn==1.0.1 필요 |
| `Tutorial_GRN__1_.py` | Gene Regulatory Network 추론 | |
| `Tutorial_Integration__2_.py` | 배치 통합 (Integration) | wandb shim 적용 |
| `Tutorial_Multiomics__1_.py` | Multi-omics 분석 | wandb shim 적용 |
| `Tutorial_Perturbation__1_.py` | 약물 반응 예측 (Perturbation) | |
| `Tutorial_Reference_Mapping__1_.py` | Reference 매핑 | |

## 환경 설정

```bash
bash setup_env.sh
```

또는 수동 설치:
```bash
pip install -r requirements.txt
```

## 호환성 패치 내용

### 1. wandb → 로컬 shim 자동 대체
`Tutorial_Annotation`, `Tutorial_Integration`, `Tutorial_Multiomics`는 원래 W&B를 사용합니다.
- wandb 로그인 없이도 실행 가능하도록 `_WandbShim`으로 자동 대체
- 실제 W&B 추적 원할 경우: `wandb login` 후 shim 제거

### 2. matplotlib 백엔드
모든 스크립트에서 `matplotlib.use('Agg')` 적용 → 헤드리스 서버에서 실행 가능  
그래프 저장: `plt.savefig('output.png')` 사용

### 3. Jupyter magic 제거
- `%matplotlib inline` → `matplotlib.use('Agg')`
- `!pip install ...` → `subprocess.run(...)`
- `%time`, `%autoreload` 등 → 주석 처리

### 4. tqdm
`tqdm.notebook` → `tqdm` (로컬 터미널 progress bar)

### 5. flash-attn
`Tutorial_Attention_GRN__1_.py`는 `flash-attn==1.0.1`이 필요합니다.  
torch==1.13.0 + CUDA 11.6 환경에서만 정상 빌드됩니다.

## 버전 정보 (poetry.lock 기준)

```
torch==1.13.0
numpy==1.21.6
pandas==1.3.5
scipy==1.7.3
anndata==0.8.0
scanpy==1.9.1
scvi-tools==0.16.4
matplotlib==3.5.2
seaborn==0.11.2
scikit-learn==1.0.2
umap-learn==0.5.3
leidenalg==0.8.10
flash-attn==1.0.1
```

## 실행

```bash
python3 Tutorial_Perturbation__1_.py
python3 Tutorial_GRN__1_.py
python3 Tutorial_Reference_Mapping__1_.py
```

> 사전 학습 모델은 `../save/scGPT_human/` 경로에 있어야 합니다 (각 스크립트의 `load_model` 참고).
