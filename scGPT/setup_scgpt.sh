#!/bin/bash
# =============================================================================
# scGPT 환경 설정 스크립트 (H200 GPU / Linux)
# Python 3.9 | PyTorch 2.1 + CUDA 12.1 | torchtext 0.16
# =============================================================================
#
# [중요] H200 GPU는 Hopper 아키텍처(sm_90)로, PyTorch >= 2.0 필수.
#        scGPT 원본의 PyTorch 1.13, flash-attn < 1.0.5 지정은 H200에서 동작 불가.
#        이 스크립트는 H200 호환 버전으로 조정합니다.
#
# 사용법:
#   chmod +x setup_scgpt.sh
#   bash setup_scgpt.sh
#
# =============================================================================
set -e

echo "============================================="
echo "  scGPT 환경 설정 시작 (H200 GPU)"
echo "============================================="

# ── 0. 시스템 확인 ─────────────────────────────────────────────────────
echo ""
echo ">>> [0/7] 시스템 환경 확인..."
echo "  Python: $(python3 --version 2>&1)"
echo "  pip:    $(pip --version 2>&1 | head -1)"

# CUDA 버전 확인
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo "  CUDA (driver): $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    echo "  [경고] nvidia-smi를 찾을 수 없습니다. GPU 환경을 확인하세요."
fi
echo ""

# ── 1. pip 업그레이드 ──────────────────────────────────────────────────
echo ">>> [1/7] pip 업그레이드..."
pip install --upgrade pip setuptools wheel

# ── 2. PyTorch 설치 (CUDA 12.1, H200 호환) ────────────────────────────
# H200(sm_90) 지원을 위해 PyTorch 2.1.2 + CUDA 12.1 사용.
# torchtext 0.16.2는 PyTorch 2.1과 매칭됩니다.
#
# 만약 시스템 CUDA가 12.4 이상이라면 아래 대안 사용 가능:
#   pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
#   pip install torchtext==0.18.0   # (PyTorch 2.3 매칭, 마지막 버전)
#
# PyTorch CUDA 런타임은 자체 포함이므로 시스템 CUDA 버전과 정확히
# 일치하지 않아도 됩니다 (드라이버만 호환되면 OK).
echo ""
echo ">>> [2/7] PyTorch 2.1.2 + CUDA 12.1 설치..."
pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 3. torchtext 설치 (PyTorch 2.1 매칭) ──────────────────────────────
# scGPT는 torchtext의 Vocab 클래스를 사용합니다.
# torchtext는 개발 중단되었으므로 버전 매칭이 중요합니다.
#   PyTorch 2.1 → torchtext 0.16.x
#   PyTorch 2.2 → torchtext 0.17.x
#   PyTorch 2.3 → torchtext 0.18.x (마지막)
echo ""
echo ">>> [3/7] torchtext 0.16.2 설치..."
pip install torchtext==0.16.2

# ── 4. 핵심 생물정보학 패키지 ──────────────────────────────────────────
echo ""
echo ">>> [4/7] 핵심 패키지 설치..."
pip install \
    numpy==1.24.4 \
    pandas==1.5.3 \
    scipy==1.10.1 \
    anndata==0.8.0 \
    scanpy==1.9.8 \
    "scvi-tools>=0.16.0,<1.0" \
    leidenalg==0.9.1 \
    "umap-learn>=0.5.3" \
    matplotlib==3.7.5 \
    "seaborn>=0.13.0" \
    "scikit-learn>=1.0.2" \
    numba \
    scikit-misc

# ── 5. scGPT 추가 의존성 (pyproject.toml 기준) ────────────────────────
# scGPT의 pyproject.toml에 명시된 의존성 중 위에서 설치하지 않은 것들
echo ""
echo ">>> [5/7] scGPT 추가 의존성 설치..."
pip install \
    datasets \
    typing-extensions \
    "scib>=1.0.3" \
    "cell-gears<0.0.3" \
    "orbax<0.1.8" \
    tqdm \
    omegaconf

# ── 6. 선택적 패키지 ──────────────────────────────────────────────────
echo ""
echo ">>> [6/7] 선택적 패키지 설치..."

# wandb (실험 추적, 선택 사항)
pip install wandb || echo "  [참고] wandb 설치 실패 — annotation.py의 wandb shim이 대체합니다."

# flash-attention 2.x (H200 sm_90 지원, 선택 사항)
# scGPT에서 fast_transformer=True 사용 시 필요.
# 설치 실패해도 annotation.py는 동작합니다 (fast_transformer=False로 변경).
echo ""
echo "  flash-attn 2.x 설치 시도 (빌드에 5~15분 소요될 수 있음)..."
pip install flash-attn --no-build-isolation 2>/dev/null \
    || echo "  [참고] flash-attn 설치 실패. annotation.py에서 fast_transformer=False로 설정하세요."

# ── 7. scGPT 설치 ─────────────────────────────────────────────────────
echo ""
echo ">>> [7/7] scGPT 설치..."
# 방법 A: pip으로 설치 (권장)
pip install scGPT

# 방법 B: 로컬 소스에서 설치 (이미 git clone한 경우)
# 프로젝트 루트(~/DW/scGPT/)에서 실행:
#   pip install -e .
# 또는 annotation.py에서 sys.path로 직접 참조 (현재 방식)


# =============================================================================
# 설치 검증
# =============================================================================
echo ""
echo "============================================="
echo "  설치 검증"
echo "============================================="

python3 -c "
import torch
print(f'  PyTorch:     {torch.__version__}')
print(f'  CUDA 사용:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU 이름:    {torch.cuda.get_device_name(0)}')
    print(f'  CUDA 버전:   {torch.version.cuda}')
    cap = torch.cuda.get_device_capability(0)
    print(f'  Compute Cap: {cap[0]}.{cap[1]}')

import torchtext
print(f'  torchtext:   {torchtext.__version__}')

import scanpy
print(f'  scanpy:      {scanpy.__version__}')

import anndata
print(f'  anndata:     {anndata.__version__}')

try:
    import scvi
    print(f'  scvi-tools:  {scvi.__version__}')
except:
    print('  scvi-tools:  설치 확인 필요')

try:
    import flash_attn
    print(f'  flash-attn:  {flash_attn.__version__}')
except:
    print('  flash-attn:  미설치 (선택사항)')

try:
    import scgpt
    print(f'  scGPT:       {scgpt.__version__}')
except:
    print('  scGPT:       로컬 소스 사용 시 정상')
"

echo ""
echo "============================================="
echo "  설치 완료!"
echo "============================================="
echo ""
echo "  실행 방법:"
echo "    cd ~/DW/scGPT"
echo "    python annotation.py"
echo ""
echo "  [참고] flash-attn 미설치 시 annotation.py에서"
echo "         fast_transformer=False 로 변경하세요."
echo ""
