#!/bin/bash
# Train NeST-VNN for a cBioPortal study
# Usage: bash train.sh <study_id> <label> <task> [cuda_id] [no_mlflow]
# Examples:
#   bash train.sh laml_tcga_pub binary_os binary
#   bash train.sh laml_tcga_pub os_months continuous
#   bash train.sh breast_msk_2025 binary_overall_survival_status binary 0 no_mlflow

set -e

STUDY_ID="${1:?Usage: bash train.sh <study_id> <label> <task> [cuda_id] [no_mlflow]}"
LABEL="${2:?Specify label column (e.g. binary_os, os_months)}"
TASK="${3:?Specify task type (binary or continuous)}"
CUDA_ID="${4:-0}"
MLFLOW_FLAG=""
if [ "${5}" = "no_mlflow" ]; then MLFLOW_FLAG="-no_mlflow"; fi
DATA_DIR="data"

NEST_DIR="${DATA_DIR}/output/${STUDY_ID}/nest_vnn_input"
MODEL_DIR="${DATA_DIR}/output/${STUDY_ID}/${LABEL}/model"

mkdir -p "${MODEL_DIR}"

echo "============================================================"
echo "Training NeST-VNN"
echo "  Study:  ${STUDY_ID}"
echo "  Label:  ${LABEL}"
echo "  Task:   ${TASK}"
echo "  Model:  ${MODEL_DIR}"
echo "============================================================"

python src/train.py \
    -onto "${NEST_DIR}/ontology.txt" \
    -gene2id "${NEST_DIR}/gene2ind.txt" \
    -cell2id "${NEST_DIR}/cell2ind.txt" \
    -train "${NEST_DIR}/training_data.txt" \
    -mutations "${NEST_DIR}/cell2mutation.txt" \
    -cn_deletions "${NEST_DIR}/cell2cndeletion.txt" \
    -cn_amplifications "${NEST_DIR}/cell2cnamplification.txt" \
    -label "${LABEL}" \
    -task "${TASK}" \
    -std "${MODEL_DIR}/std.txt" \
    -model "${MODEL_DIR}" \
    -genotype_hiddens 4 \
    -lr 0.0001 \
    -cuda "${CUDA_ID}" \
    -epoch 200 \
    -batchsize 512 \
    -optimize 1 \
    -zscore_method auc \
    ${MLFLOW_FLAG}