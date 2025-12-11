#!/bin/bash
# Roofline profiling with Nsight Compute
# Three-stage KV cache analysis
#
# Usage: sudo bash run_ncu.sh [--P 512] [--batch 1] [--model gpt2-medium]

set -e

# Defaults
P=512           # Prompt length (prefill tokens)
BATCH_SIZE=1
MODEL="gpt2-medium"
DTYPE="float16"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --P) P="$2"; shift 2 ;;
        --batch) BATCH_SIZE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

OUTPUT_DIR="benchmark/roofline/ncu_reports"
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "KV Cache Roofline Analysis"
echo "Model: ${MODEL}, Prompt: ${P} tokens, Batch: ${BATCH_SIZE}"
echo "=============================================="

echo ""
echo "=== Stage 1/3: V0 Prefill (no cache) ==="
/local/kyle0/conda_envs/nanogpt/bin/ncu -f --set detailed \
    --profile-from-start off \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    -o ${OUTPUT_DIR}/v0_prefill_P${P}_B${BATCH_SIZE}_${MODEL} \
    /local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/profile_decode_step.py \
        --version v0_prefill --model $MODEL --P $P --batch_size $BATCH_SIZE --dtype $DTYPE

echo ""
echo "=== Stage 2/3: V1 Prefill (build cache) ==="
/local/kyle0/conda_envs/nanogpt/bin/ncu -f --set detailed \
    --profile-from-start off \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    -o ${OUTPUT_DIR}/v1_prefill_P${P}_B${BATCH_SIZE}_${MODEL} \
    /local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/profile_decode_step.py \
        --version v1_prefill --model $MODEL --P $P --batch_size $BATCH_SIZE --dtype $DTYPE

echo ""
echo "=== Stage 3/3: V1 Decode (use cache) ==="
/local/kyle0/conda_envs/nanogpt/bin/ncu -f --set detailed \
    --profile-from-start off \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    -o ${OUTPUT_DIR}/v1_decode_P${P}_B${BATCH_SIZE}_${MODEL} \
    /local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/profile_decode_step.py \
        --version v1_decode --model $MODEL --P $P --batch_size $BATCH_SIZE --dtype $DTYPE

echo ""
echo "=== Done ==="
echo "Reports saved to ${OUTPUT_DIR}/"
echo ""
echo "Expected results:"
echo "  - v0_prefill & v1_prefill: Similar (both process ${P} tokens)"
echo "  - v1_decode: Much lower FLOPs, lower arithmetic intensity (memory-bound)"
