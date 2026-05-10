#!/bin/bash
# Annotate hierarchy with RLIPP explainability scores
# Usage: bash annotate.sh <study_id> <label> <task> [cpu_count]
# Examples:
#   bash annotate.sh laml_tcga_pub binary_os binary 4
#   bash annotate.sh breast_msk_2025 binary_overall_survival_status binary 8

set -e

STUDY_ID="${1:?Usage: bash annotate.sh <study_id> <label> <task> [cpu_count]}"
LABEL="${2:?Specify label column}"
TASK="${3:?Specify task type (binary or continuous)}"
CPU_COUNT="${4:-4}"

echo "============================================================"
echo "Annotating NeST-VNN Hierarchy"
echo "  Study:  ${STUDY_ID}"
echo "  Label:  ${LABEL}"
echo "  Task:   ${TASK}"
echo "============================================================"

python src/annotate_hierarchy.py \
    "${STUDY_ID}" \
    -label "${LABEL}" \
    -task "${TASK}" \
    -cpu_count "${CPU_COUNT}" \
    -genotype_hiddens 4