# 환경 생성
conda create -n scbert python=3.10 -y
conda activate scbert

# conda 패키지
conda install -y \
  numpy=1.24.4 \
  pandas=1.5.3 \
  scikit-learn=1.3.2 \
  matplotlib=3.7.5 \
  scanpy=1.9.8 \
  anndata=0.9.8

# PyTorch (H200 / CUDA 12.1)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# pip 패키지
pip install scipy==1.11.4
pip install transformers==4.30.2
pip install einops==0.7.0

# libtiff
conda install -c conda-forge libtiff=4.6.0 -y

# Pillow
pip uninstall -y pillow
pip install pillow==10.0.1

# 버전 유틸
pip uninstall -y get-version legacy-api-wrap
pip install legacy-api-wrap==1.2 get-version==2.1

# scBERT 핵심
pip install local-attention==1.9.0
pip install performer-pytorch==1.1.4
pip install product-key-memory==0.2.1

# 기존
from torch.optim.lr_scheduler import _LRScheduler
class CosineAnnealingWarmupRestarts(_LRScheduler):

# 수정
from torch.optim.lr_scheduler import LRScheduler
class CosineAnnealingWarmupRestarts(LRScheduler):

# CUDA 확인
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# performer_pytorch import 확인 (가장 중요)
python -c "from performer_pytorch import PerformerLM; print('OK')"

torchrun --nproc_per_node=<GPU수> --master_port=29500 pretrain.py \
  --data_path ./data/panglao_human.h5ad \
  --ckpt_dir ./ckpts/ \
  --model_name panglao_pretrain \
  --batch_size 3 \
  --epoch 100
