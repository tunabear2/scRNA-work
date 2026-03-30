#!/bin/bash
# =============================================================================
# scGPT 환경 설정 스크립트
# pip_conda.txt 기준 | PyTorch 2.1.2+cu121 | flash-attn 2.7.3
# =============================================================================
#
# 사용법:
#   conda create -n scgpt python=3.11 -y && conda activate scgpt
#   bash setup_scgpt.sh
#
# =============================================================================
set -e

echo "============================================="
echo "  scGPT 환경 설정 시작"
echo "============================================="

# ── 시스템 확인 ────────────────────────────────────────────────────────────
echo ""
echo ">>> [0] 시스템 환경 확인..."
echo "  Python: $(python --version 2>&1)"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  GPU: /'
else
    echo "  [경고] nvidia-smi 없음 — GPU 환경을 확인하세요."
fi

pip install --upgrade pip setuptools wheel

# ── 1. PyTorch 2.1.2 + CUDA 12.1 ──────────────────────────────────────────
echo ""
echo ">>> [1] PyTorch 2.1.2 + CUDA 12.1 설치..."
pip install \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install \
    torchtext==0.16.2 \
    torchdata==0.7.1 \
    torchmetrics==1.8.2

# ── 2. PyTorch Geometric (pt21cu121 빌드) ─────────────────────────────────
# torch-scatter / torch-sparse / torch-cluster 는 PyTorch 버전에 맞는
# 전용 휠을 pyg.org 에서 받아야 합니다.
echo ""
echo ">>> [2] PyTorch Geometric 설치..."
pip install \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    torch-cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-geometric==2.6.1

# ── 3. 핵심 과학/바이오 패키지 ────────────────────────────────────────────
echo ""
echo ">>> [3] 핵심 패키지 설치..."
pip install \
    numpy==1.24.4 \
    scipy==1.10.1 \
    pandas==2.3.3 \
    anndata==0.8.0 \
    scanpy==1.9.8 \
    scvi-tools==0.20.3 \
    scib==1.1.7 \
    mudata==0.2.3 \
    h5py==3.14.0

# ── 4. ML 유틸리티 ────────────────────────────────────────────────────────
echo ""
echo ">>> [4] ML 유틸리티 설치..."
pip install \
    scikit-learn==1.6.1 \
    scikit-misc==0.3.1 \
    umap-learn==0.5.11 \
    numba==0.60.0 \
    leidenalg==0.11.0 \
    igraph==1.0.0 \
    pynndescent==0.6.0 \
    pytorch-lightning==1.9.5

# ── 5. 시각화/분석 ────────────────────────────────────────────────────────
echo ""
echo ">>> [5] 시각화/분석 패키지 설치..."
pip install \
    matplotlib==3.9.4 \
    seaborn==0.13.2 \
    networkx==3.2.1 \
    pydot==4.0.1 \
    statsmodels==0.14.6 \
    gseapy==1.1.12

# ── 6. scGPT 의존성 ───────────────────────────────────────────────────────
echo ""
echo ">>> [6] scGPT 의존성 설치..."
pip install \
    datasets==2.21.0 \
    omegaconf==2.3.0 \
    einops==0.8.2 \
    local-attention==1.11.2 \
    cell-gears==0.0.2 \
    orbax==0.1.7 \
    natsort==8.4.0 \
    tqdm==4.67.3 \
    wandb==0.25.1 \
    dcor==0.6

# ── 7. faiss (유사도 검색, reference_mapping.py 필요) ─────────────────────
echo ""
echo ">>> [7] faiss 설치..."
pip install faiss-gpu==1.9.0 \
    || pip install faiss-cpu==1.9.0 \
    || echo "  [참고] faiss 설치 실패 — reference_mapping.py의 CellXGene 모드에서만 필요합니다."

# ── 8. flash-attn 2.7.3 (선택, H200 호환) ─────────────────────────────────
# fast_transformer=True 사용 시 필요. 미설치 시 자동으로 표준 attention으로 동작.
# 빌드에 5~15분 소요될 수 있음.
echo ""
echo ">>> [8] flash-attn 2.7.3 설치 (빌드 중, 시간 소요)..."
pip install flash-attn==2.7.3 --no-build-isolation \
    || echo "  [참고] flash-attn 설치 실패 — 스크립트에서 fast_transformer=False 로 설정하세요."

# ── 9. scGPT 로컬 설치 ────────────────────────────────────────────────────
echo ""
echo ">>> [9] scGPT 로컬 소스 설치..."
# 프로젝트 루트에서 실행 시 로컬 scgpt/ 패키지를 editable 모드로 설치
pip install -e . \
    || echo "  [참고] setup.py/pyproject.toml 없으면 스크립트에서 직접 import됩니다."

# ── 검증 ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  설치 검증"
echo "============================================="

python -c "
import torch
print(f'  PyTorch:          {torch.__version__}')
print(f'  CUDA 사용 가능:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:              {torch.cuda.get_device_name(0)}')
    print(f'  Compute Cap:      {torch.cuda.get_device_capability(0)}')

import torchtext;       print(f'  torchtext:        {torchtext.__version__}')
import scanpy;          print(f'  scanpy:           {scanpy.__version__}')
import anndata;         print(f'  anndata:          {anndata.__version__}')
import scvi;            print(f'  scvi-tools:       {scvi.__version__}')

try:
    import scgpt;       print(f'  scGPT:            {scgpt.__version__}')
except Exception as e:  print(f'  scGPT:            {e}')

try:
    import flash_attn;  print(f'  flash-attn:       {flash_attn.__version__}')
except:                 print('  flash-attn:       미설치 (선택사항)')

try:
    import faiss;       print(f'  faiss:            설치됨')
except:                 print('  faiss:            미설치 (reference_mapping 일부 기능 제한)')
"

echo ""
echo "============================================="
echo "  완료!"
echo "============================================="
echo ""
echo "  실행:"
echo "    cd ~/DW/scGPT && python annotation.py"
echo ""
