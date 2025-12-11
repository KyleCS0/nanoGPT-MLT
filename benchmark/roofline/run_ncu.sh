#!/bin/bash
# Roofline profiling with Nsight Compute
# Run on A6000

set -e

T=1024
BATCH_SIZE=16
MODEL="gpt2-xl"
DTYPE="float16"
OUTPUT_DIR="benchmark/roofline/ncu_reports"
mkdir -p $OUTPUT_DIR

echo "=== Profiling V0 (no cache) [${MODEL}, T=${T}, B=${BATCH_SIZE}] ==="
/local/kyle0/conda_envs/nanogpt/bin/ncu --set detailed \
    --profile-from-start off \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    -o ${OUTPUT_DIR}/v0_T${T}_B${BATCH_SIZE}_${MODEL} \
    /local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/profile_decode_step.py --version v0 --model $MODEL --T $T --batch_size $BATCH_SIZE --dtype $DTYPE

echo "=== Profiling V1 (KV-cache) [${MODEL}, T=${T}, B=${BATCH_SIZE}] ==="
/local/kyle0/conda_envs/nanogpt/bin/ncu --set detailed \
    --profile-from-start off \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    -o ${OUTPUT_DIR}/v1_T${T}_B${BATCH_SIZE}_${MODEL} \
    /local/kyle0/conda_envs/nanogpt/bin/python benchmark/roofline/profile_decode_step.py --version v1 --model $MODEL --T $T --batch_size $BATCH_SIZE --dtype $DTYPE

echo "=== Done ==="
echo "Reports saved to ${OUTPUT_DIR}/"
echo "View with: ncu-ui ${OUTPUT_DIR}/v0_T${T}.ncu-rep"
