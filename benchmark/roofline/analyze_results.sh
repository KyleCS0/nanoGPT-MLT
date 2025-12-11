#!/bin/bash
# Analyze NCU reports and generate roofline plot
# Three-stage KV cache analysis
#
# Usage: bash analyze_results.sh [--P 512] [--batch 1] [--model gpt2-medium]

set -e

# Defaults (must match run_ncu.sh)
P=512
BATCH_SIZE=1
MODEL="gpt2-medium"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --P) P="$2"; shift 2 ;;
        --batch) BATCH_SIZE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

REPORT_DIR="benchmark/roofline/ncu_reports"

# Check if NCU reports exist
if [ ! -f "${REPORT_DIR}/v0_prefill_P${P}_B${BATCH_SIZE}_${MODEL}.ncu-rep" ]; then
    echo "Error: NCU reports not found."
    echo "Please run: sudo bash benchmark/roofline/run_ncu.sh --P $P --batch $BATCH_SIZE --model $MODEL"
    exit 1
fi

echo "=============================================="
echo "Parsing NCU Reports"
echo "=============================================="

echo ""
echo "Parsing V0 Prefill..."
/local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/parse_ncu_results.py \
    --report ${REPORT_DIR}/v0_prefill_P${P}_B${BATCH_SIZE}_${MODEL}.ncu-rep \
    --output ${REPORT_DIR}/v0_prefill_metrics.json

echo ""
echo "Parsing V1 Prefill..."
/local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/parse_ncu_results.py \
    --report ${REPORT_DIR}/v1_prefill_P${P}_B${BATCH_SIZE}_${MODEL}.ncu-rep \
    --output ${REPORT_DIR}/v1_prefill_metrics.json

echo ""
echo "Parsing V1 Decode..."
/local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/parse_ncu_results.py \
    --report ${REPORT_DIR}/v1_decode_P${P}_B${BATCH_SIZE}_${MODEL}.ncu-rep \
    --output ${REPORT_DIR}/v1_decode_metrics.json

echo ""
echo "Generating Roofline Plot..."
/local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/plot_roofline.py \
    --v0_prefill ${REPORT_DIR}/v0_prefill_metrics.json \
    --v1_prefill ${REPORT_DIR}/v1_prefill_metrics.json \
    --v1_decode ${REPORT_DIR}/v1_decode_metrics.json \
    --output benchmark/roofline/roofline.png \
    --model ${MODEL} \
    --batch-size ${BATCH_SIZE} \
    --seq-length ${P}

echo ""
echo "Done. Results in benchmark/roofline/roofline.png and roofline.pdf"
