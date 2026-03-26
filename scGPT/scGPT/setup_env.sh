#!/bin/bash
# scGPT 환경 설정 스크립트 (Linux)
# Python 3.8 이상 필요

set -e

echo "=== scGPT 환경 설정 시작 ==="

# 1. Python 버전 확인
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
python 3.9 기준.
# 2. pip 업그레이드
pip install --upgrade pip

# 3. PyTorch 설치 (CUDA 11.6 기준, poetry.lock: torch==1.13.0)
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
# CPU only인 경우:
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
# 3. PyTorch 설치 (CUDA 11.6 기준, poetry.lock: torch==1.13.0)
echo ">>> PyTorch 설치 (torch==1.13.0)..."

# 4. 핵심 패키지 설치
echo ">>> 핵심 패키지 설치..."
pip install \
    numpy==1.21.6 \
    pandas==1.3.5 \
    scipy==1.7.3 \
    anndata==0.8.0 \
    scanpy==1.9.1 \
    leidenalg==0.8.10 \
    "umap-learn==0.5.3" \
    matplotlib==3.5.2 \
    seaborn==0.11.2 \
    "scikit-learn==1.0.2" \
    tqdm \
    omegaconf \
    wandb

# 5. scvi-tools (주의: 0.16.4는 jax 없는 버전)
echo ">>> scvi-tools 설치..."
pip install "scvi-tools==0.16.4"

# 6. flash-attn (Attention_GRN 튜토리얼에만 필요)
# torch==1.13.0 + CUDA 11.6 환경에서 빌드 필요
echo ">>> flash-attn 설치 (Attention_GRN 튜토리얼에만 필요)..."
pip install "flash-attn==1.0.1" || echo "[WARNING] flash-attn 설치 실패 - Attention_GRN 튜토리얼은 이 패키지가 필요합니다."

# 7. scGPT 소스 설치
pip install scGPT==0.2.4

echo ""
echo "=== 환경 설정 완료 ==="
echo "실행 예시:"
echo "  python3 Tutorial_Annotation__2_.py"
echo "  python3 Tutorial_Perturbation__1_.py"
