#!/bin/bash
# Predict with a trained NeST-VNN model
# Usage: bash predict.sh <study_id> <label> <task> [cuda_id]
# Examples:
#   bash predict.sh laml_tcga_pub binary_os binary
#   bash predict.sh breast_msk_2025 binary_overall_survival_status binary

set -e

STUDY_ID="${1:?Usage: bash predict.sh <study_id> <label> <task> [cuda_id]}"
LABEL="${2:?Specify label column}"
TASK="${3:?Specify task type (binary or continuous)}"
CUDA_ID="${4:-0}"
DATA_DIR="data"

NEST_DIR="${DATA_DIR}/output/${STUDY_ID}/nest_vnn_input"
MODEL_DIR="${DATA_DIR}/output/${STUDY_ID}/model"
METRICS_DIR="${DATA_DIR}/output/${STUDY_ID}/metrics"

mkdir -p "${METRICS_DIR}/hidden"

echo "============================================================"
echo "Predicting with NeST-VNN"
echo "  Study:   ${STUDY_ID}"
echo "  Label:   ${LABEL}"
echo "  Task:    ${TASK}"
echo "  Model:   ${MODEL_DIR}/model_final.pt"
echo "  Output:  ${METRICS_DIR}"
echo "============================================================"

python src/predict.py \
    -predict "${NEST_DIR}/training_data.txt" \
    -gene2id "${NEST_DIR}/gene2ind.txt" \
    -cell2id "${NEST_DIR}/cell2ind.txt" \
    -mutations "${NEST_DIR}/cell2mutation.txt" \
    -cn_deletions "${NEST_DIR}/cell2cndeletion.txt" \
    -cn_amplifications "${NEST_DIR}/cell2cnamplification.txt" \
    -fusions "${NEST_DIR}/cell2fusion.txt" \
    -label "${LABEL}" \
    -task "${TASK}" \
    -std "${MODEL_DIR}/std.txt" \
    -load "${MODEL_DIR}/model_final.pt" \
    -hidden "${METRICS_DIR}/hidden" \
    -result "${METRICS_DIR}/predict" \
    -cuda "${CUDA_ID}" \
    -zscore_method auc \
    -batchsize 64