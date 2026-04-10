"""
04. Gene Perturbation (GEARS)
데이터: datasets/CellFM/norman.h5ad (GEARS 포맷으로 자동 변환)
출력:  results/04_gene_perturbation/  (모델, 평가 결과)
       results/04_gene_perturbation/metrics.json
"""
import os, sys, datetime
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from log_utils.logger import get_logger, save_metrics

TASK        = "04_gene_perturbation"
DATA_DIR    = os.path.join(ROOT_DIR, "datasets", "CellFM")
RESULT_DIR  = os.path.join(ROOT_DIR, "results", TASK)

os.makedirs(RESULT_DIR, exist_ok=True)
logger = get_logger(TASK, RESULT_DIR)


def prepare_gears_data(data_dir):
    """
    norman.h5ad를 GEARS PertData 형식으로 로드.
    pip gears는 data_name='norman' 으로 figshare에서 직접 다운로드하거나
    이미 준비된 폴더를 data_path 로 지정해야 한다.

    여기서는 로컬 norman.h5ad를 GEARS 형식 폴더로 변환한다.
    """
    import pickle, scanpy as sc
    from pathlib import Path

    src = os.path.join(data_dir, "norman.h5ad")
    dst_dir = os.path.join(data_dir, "norman")
    dst_pkl = os.path.join(dst_dir, "perturb_processed.h5ad")

    if os.path.exists(dst_pkl):
        logger.info(f"캐시 데이터 사용: {dst_dir}")
        return dst_dir

    os.makedirs(dst_dir, exist_ok=True)
    logger.info(f"norman.h5ad → GEARS 포맷 변환: {dst_dir}")

    adata = sc.read_h5ad(src)
    logger.info(f"로드: shape={adata.shape}, obs={list(adata.obs.columns)}")

    # GEARS 필수 obs 컬럼 확인 및 정규화
    # condition: 'CTRL', 'GeneA+ctrl', 'GeneA+GeneB' 형태
    if "condition" not in adata.obs.columns:
        raise ValueError("norman.h5ad에 'condition' 컬럼이 없습니다.")
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "K562"   # norman 데이터셋 기본 세포주

    # GEARS가 읽을 수 있는 h5ad 저장
    adata.write_h5ad(dst_pkl)
    logger.info(f"변환 완료: {dst_pkl}")
    return dst_dir


def main():
    from gears import PertData, GEARS

    # ── 하이퍼파라미터 ────────────────────────────────────
    split               = "simulation"
    seed                = 3
    epochs              = 15
    batch_size          = 10
    accumulation_steps  = 5
    test_batch_size     = 32
    hidden_size         = 512
    train_gene_set_size = 0.75
    highres             = 0
    lr                  = 0.005
    device              = "cuda:0"

    logger.info("GEARS 데이터 준비 중...")
    gears_data_dir = prepare_gears_data(DATA_DIR)

    # PertData 로드
    pert_data = PertData(DATA_DIR)
    try:
        pert_data.load(data_path=gears_data_dir)
        logger.info("로컬 데이터 로드 성공")
    except Exception as e:
        logger.warning(f"로컬 로드 실패: {e}")
        logger.info("figshare에서 norman 다운로드 시도...")
        pert_data = PertData(DATA_DIR)
        pert_data.load(data_name="norman")

    pert_data.prepare_split(split=split, seed=seed,
                             train_gene_set_size=train_gene_set_size)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)
    logger.info("데이터 로더 준비 완료")

    gears_model = GEARS(pert_data, device=device)
    gears_model.model_initialize(
        hidden_size=hidden_size,
    )
    logger.info(f"모델 초기화 완료. 학습 시작 ({epochs} epoch)")
    gears_model.train(epochs=epochs, lr=lr)
    logger.info(f"학습 완료. 결과: {RESULT_DIR}")

    # ── 평가 ─────────────────────────────────────────────
    from gears.inference import evaluate, compute_metrics

    test_loader   = pert_data.dataloader["test_loader"]
    test_res      = evaluate(test_loader, gears_model.best_model,
                             gears_model.config['uncertainty'], gears_model.device)
    test_metrics, test_pert_res = compute_metrics(test_res)
    logger.info(f"test_metrics: {test_metrics}")

    metrics = {
        "task":       TASK,
        "dataset":    "norman",
        "epochs":     epochs,
        "split":      split,
        "mse":        float(test_metrics.get("mse", -1)),
        "mse_de":     float(test_metrics.get("mse_de", -1)),
        "pearson":    float(test_metrics.get("pearson", -1)),
        "pearson_de": float(test_metrics.get("pearson_de", -1)),
        "timestamp":  datetime.datetime.now().isoformat(),
        "result_dir": RESULT_DIR,
    }
    path = save_metrics(metrics, RESULT_DIR)
    logger.info(f"결과 저장: {path}")


if __name__ == "__main__":
    main()
