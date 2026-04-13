"""
04. Gene Perturbation (GEARS)
데이터: datasets/CellFM/norman.h5ad (GEARS 포맷으로 자동 변환)
출력:  results/04_gene_perturbation/
         metrics.json
         predicted_expression.csv  (평균 예측 발현량 per perturbation)
         figures/scatter_pred_vs_truth.png
         figures/top_perturbations_pearson.png
"""
import os, sys, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from log_utils.logger import get_logger, save_metrics

TASK        = "04_gene_perturbation"
DATA_DIR    = os.path.join(ROOT_DIR, "datasets", "CellFM")
RESULT_DIR  = os.path.join(ROOT_DIR, "results", TASK)
FIG_DIR     = os.path.join(RESULT_DIR, "figures")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
logger = get_logger(TASK, RESULT_DIR)


def prepare_gears_data(data_dir):
    """
    norman.h5ad를 GEARS PertData 형식으로 로드.
    pip gears는 data_name='norman' 으로 figshare에서 직접 다운로드하거나
    이미 준비된 폴더를 data_path 로 지정해야 한다.

    여기서는 로컬 norman.h5ad를 GEARS 형식 폴더로 변환한다.
    """
    import scanpy as sc

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

    if "condition" not in adata.obs.columns:
        raise ValueError("norman.h5ad에 'condition' 컬럼이 없습니다.")
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "K562"

    adata.write_h5ad(dst_pkl)
    logger.info(f"변환 완료: {dst_pkl}")
    return dst_dir


def save_figures(test_res, gene_names=None):
    """예측 발현값 scatter plot 및 Pearson bar chart 저장."""
    logger.info("Figure 생성 시작...")

    # ── 예측 발현값 수집 (perturbation별 평균) ────────────────────
    rows = []
    pred_means, truth_means, cond_labels = [], [], []

    for cond, vals in test_res.items():
        try:
            pred  = np.array(vals.get("pred",  vals.get("pred_de",  [])))
            truth = np.array(vals.get("truth", vals.get("truth_de", [])))
            if pred.ndim == 0 or truth.ndim == 0:
                continue
            if pred.ndim == 1:
                pred  = pred.reshape(1, -1)
                truth = truth.reshape(1, -1)
            pm = pred.mean(axis=0)
            tm = truth.mean(axis=0)
            pred_means.append(pm)
            truth_means.append(tm)
            cond_labels.append(str(cond))
            for g_idx, (p, t) in enumerate(zip(pm, tm)):
                rows.append({
                    "condition": str(cond),
                    "gene_idx":  g_idx,
                    "gene":      gene_names[g_idx] if gene_names and g_idx < len(gene_names) else str(g_idx),
                    "pred_mean": float(p),
                    "truth_mean":float(t),
                })
        except Exception:
            continue

    if not rows:
        logger.warning("예측 발현값이 비어있어 figure를 생성하지 않습니다.")
        return

    # 예측 발현값 CSV 저장
    df_pred = pd.DataFrame(rows)
    csv_path = os.path.join(RESULT_DIR, "predicted_expression.csv")
    df_pred.to_csv(csv_path, index=False)
    logger.info(f"predicted_expression.csv 저장: {len(df_pred)} rows")

    # ── 1. Scatter: 전체 (truth_mean vs pred_mean) ────────────────
    all_truth = np.concatenate(truth_means)
    all_pred  = np.concatenate(pred_means)
    # 너무 많으면 subsample
    if len(all_truth) > 50000:
        idx = np.random.choice(len(all_truth), 50000, replace=False)
        all_truth, all_pred = all_truth[idx], all_pred[idx]

    corr = np.corrcoef(all_truth, all_pred)[0, 1] if len(all_truth) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_truth, all_pred, s=1, alpha=0.3, rasterized=True)
    mn, mx = min(all_truth.min(), all_pred.min()), max(all_truth.max(), all_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="y=x")
    ax.set_xlabel("True Mean Expression")
    ax.set_ylabel("Predicted Mean Expression")
    ax.set_title(f"Predicted vs Truth  (r={corr:.3f})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scatter_pred_vs_truth.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("scatter_pred_vs_truth.png 저장")

    # ── 2. Pearson per perturbation (상위 30개) ───────────────────
    pearson_list = []
    for pm, tm, cond in zip(pred_means, truth_means, cond_labels):
        if len(pm) < 2:
            continue
        r = np.corrcoef(tm, pm)[0, 1]
        if not np.isnan(r):
            pearson_list.append((cond, r))

    if pearson_list:
        pearson_list.sort(key=lambda x: x[1], reverse=True)
        top = pearson_list[:30]
        conds_top = [c for c, _ in top]
        rs_top    = [r for _, r in top]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(conds_top)), rs_top, color="steelblue")
        ax.set_xticks(range(len(conds_top)))
        ax.set_xticklabels(conds_top, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Pearson r")
        ax.set_title("Top-30 Perturbations by Pearson Correlation")
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "top_perturbations_pearson.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("top_perturbations_pearson.png 저장")


def main():
    from gears import PertData, GEARS

    # ── 하이퍼파라미터 ────────────────────────────────────────
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
    gears_model.model_initialize(hidden_size=hidden_size)
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

    # gene names 수집
    gene_names = list(pert_data.gene_names) if hasattr(pert_data, "gene_names") else None

    # Figure 저장
    save_figures(test_res, gene_names)

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
