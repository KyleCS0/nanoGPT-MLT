#!/bin/bash
set -e
T=1024
MODEL="gpt2-medium"
REPORT_DIR="benchmark/roofline/ncu_reports"
mkdir -p benchmark/roofline/metrics

echo "Parsing V0..."
/local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/parse_ncu_results.py \
    --report ${REPORT_DIR}/v0_T${T}_${MODEL}.ncu-rep \
    --output ${REPORT_DIR}/v0_metrics.json

echo "Parsing V1..."
/local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/parse_ncu_results.py \
    --report ${REPORT_DIR}/v1_T${T}_${MODEL}.ncu-rep \
    --output ${REPORT_DIR}/v1_metrics.json

echo "Generating Plot..."
/local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/plot_roofline.py \
    --v0 benchmark/roofline/metrics/v0.json \
    --v1 benchmark/roofline/metrics/v1.json \
    --output benchmark/roofline/roofline.png

echo "Done. Results in benchmark/roofline/roofline.png"
