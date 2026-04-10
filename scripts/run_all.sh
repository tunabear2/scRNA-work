#!/bin/bash
# 모든 실습 태스크를 순서대로 실행
# 사용: bash scripts/run_all.sh [태스크번호]
#       bash scripts/run_all.sh        # 전체 실행
#       bash scripts/run_all.sh 1      # 01번만 실행

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="scfm"

cd "$ROOT_DIR"

run_task() {
    local num="$1"
    local name="$2"
    local script="$3"
    echo ""
    echo "════════════════════════════════════════"
    echo "  태스크 ${num}: ${name}"
    echo "════════════════════════════════════════"
    conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/$script"
}

TARGET="${1:-all}"

if [[ "$TARGET" == "all" || "$TARGET" == "1" ]]; then
    run_task "01" "Cell Annotation (Finetune)" "01_cell_annotation_finetune.py"
fi
if [[ "$TARGET" == "all" || "$TARGET" == "2" ]]; then
    run_task "02" "Cell Annotation (Zero-shot)" "02_cell_annotation_zeroshot.py"
fi
if [[ "$TARGET" == "all" || "$TARGET" == "3" ]]; then
    run_task "03" "Batch Integration" "03_batch_integration.py"
fi
if [[ "$TARGET" == "all" || "$TARGET" == "4" ]]; then
    run_task "04" "Gene Perturbation (GEARS)" "04_gene_perturbation.py"
fi
if [[ "$TARGET" == "all" || "$TARGET" == "5" ]]; then
    run_task "05" "Binary Gene Function" "05_binary_gene_function.py"
fi
if [[ "$TARGET" == "all" || "$TARGET" == "6" ]]; then
    run_task "06" "Multiclass Gene Function (GO)" "06_multiclass_gene_function.py"
fi
if [[ "$TARGET" == "all" || "$TARGET" == "7" ]]; then
    run_task "07" "lncRNA 동정" "07_lncrna.py"
fi

echo ""
echo "════════════════════════════════════════"
echo "  모든 태스크 완료"
echo "════════════════════════════════════════"
