#!/bin/bash
# scBERT 환경 설치 스크립트
# Python 3.10 + CUDA 12.1 기반
# 사전 조건: miniconda(또는 anaconda) 설치 완료

set -e

ENV_NAME="scbert"

echo "=== scBERT 환경 설치 시작 ==="

# 1. conda 환경 생성
echo "[1/3] conda 환경 생성: $ENV_NAME (Python 3.10)"
conda create -y -n $ENV_NAME python=3.10 \
    numpy=1.24.4 \
    pandas=1.5.3 \
    packaging \
    python-dateutil \
    pytz \
    six \
    setuptools \
    wheel \
    threadpoolctl \
    -c conda-forge -c defaults

# 2. conda 환경 활성화
echo "[2/3] 패키지 설치"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# PyTorch (CUDA 12.1)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 triton==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Performer / Attention 관련
pip install performer-pytorch==1.1.4 \
    local-attention==1.9.0 \
    axial-positional-embedding==0.3.12 \
    product-key-memory==0.2.1 \
    einops==0.7.0

# 단일세포 분석
pip install scanpy==1.9.8 \
    anndata==0.9.2 \
    h5py==3.16.0 \
    umap-learn==0.5.12 \
    numba==0.65.0 \
    llvmlite==0.47.0 \
    pynndescent==0.6.0

# 머신러닝 / 수치
pip install scikit-learn==1.3.2 \
    scipy==1.11.4 \
    statsmodels==0.14.6 \
    patsy==1.0.2

# NLP / Transformers
pip install transformers==4.30.2 \
    tokenizers==0.13.3 \
    huggingface-hub==0.36.2 \
    safetensors==0.7.0

# 시각화
pip install matplotlib==3.7.5 \
    seaborn==0.13.2

# 기타 유틸
pip install tqdm==4.67.3 \
    pyyaml==6.0.3 \
    requests==2.28.1 \
    natsort==8.4.0 \
    legacy-api-wrap==1.2 \
    get-version==2.1 \
    session-info==1.0.1 \
    stdlib-list==0.12.0

echo "[3/3] 설치 검증"
python - <<'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
import scanpy; print(f"scanpy: {scanpy.__version__}")
import performer_pytorch; print("performer-pytorch: OK")
print("=== 설치 완료 ===")
EOF
