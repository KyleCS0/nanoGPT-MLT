#!/bin/bash
# =============================================================================
# Full Benchmark Suite for nanoGPT KV-cache Optimizations
# =============================================================================
#
# Runs all benchmarks: latency, VRAM, per-phase timing, and perplexity.
# Optionally runs roofline profiling (requires Nsight Compute).
#
# MUST be run with sudo for GPU clock control.
#
# Usage:
#   sudo ./benchmark/run_full_benchmark.sh [OPTIONS]
#
# Options:
#   --clean           Remove results.jsonl and plots/ before running
#   --versions "..."  Space-separated versions (default: "v0 v1")
#                     Available: v0 (no cache), v1 (KV-cache), v2 (INT8),
#                                v3 (cross-layer), v4 (INT8+cross-layer)
#   --config FILE     Config file (default: benchmark/config.yaml)
#   --skip-benchmark  Only regenerate plots from existing results
#   --skip-perplexity Skip perplexity evaluation (slow)
#   --roofline        Run Nsight Compute roofline profiling (very slow)
#   --help            Show this help
#
# Examples:
#   sudo ./benchmark/run_full_benchmark.sh --clean --versions "v0 v1 v2 v3 v4"
#   sudo ./benchmark/run_full_benchmark.sh --skip-benchmark
#   sudo ./benchmark/run_full_benchmark.sh --versions "v1 v2" --skip-perplexity
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# =============================================================================
# Default Settings
# =============================================================================
CLEAN=false
VERSIONS="v0 v1 v2 v3 v4"
SKIP_BENCHMARK=false
SKIP_PERPLEXITY=false
RUN_ROOFLINE=false
CONFIG="benchmark/config.yaml"

# =============================================================================
# Argument Parsing
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --versions)
            VERSIONS="$2"
            shift 2
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --skip-perplexity)
            SKIP_PERPLEXITY=true
            shift
            ;;
        --roofline)
            RUN_ROOFLINE=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            head -35 "$0" | tail -32
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Output Formatting
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${CYAN}${BOLD}=== $1 ===${NC}\n"; }

# =============================================================================
# Header
# =============================================================================
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  nanoGPT KV-Cache Benchmark Suite${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo "  Project root:    $PROJECT_ROOT"
echo "  Config file:     $CONFIG"
echo "  Versions:        $VERSIONS"
echo "  Clean first:     $CLEAN"
echo "  Skip benchmark:  $SKIP_BENCHMARK"
echo "  Skip perplexity: $SKIP_PERPLEXITY"
echo "  Run roofline:    $RUN_ROOFLINE"
echo ""
echo -e "${BOLD}============================================================${NC}"

# =============================================================================
# Permission Check
# =============================================================================
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run with sudo for GPU clock control"
    echo "  Usage: sudo $0 [OPTIONS]"
    exit 1
fi

# Get actual user (not root) for Python execution
ACTUAL_USER="${SUDO_USER:-$USER}"
log_info "Running as root, Python will execute as: $ACTUAL_USER"

run_as_user() {
    sudo -u "$ACTUAL_USER" "$@"
}

# =============================================================================
# Cleanup Handler (always releases GPU clocks on exit)
# =============================================================================
cleanup() {
    echo ""
    log_info "Releasing GPU clock lock..."
    nvidia-smi -rgc 2>/dev/null || true
    log_success "GPU clocks reset to default"
}
trap cleanup EXIT

# =============================================================================
# Clean Previous Results (if requested)
# =============================================================================
if [ "$CLEAN" = true ]; then
    log_section "Cleaning Previous Results"
    rm -f benchmark/results.jsonl
    rm -rf benchmark/plots/*
    log_success "Removed benchmark/results.jsonl and benchmark/plots/*"
fi

# Create output directories
mkdir -p benchmark/plots

# =============================================================================
# Lock GPU Clocks
# =============================================================================
log_section "Locking GPU Clocks"

nvidia-smi -pm 1 2>/dev/null || log_warn "Could not enable persistence mode"

# Try common clock frequencies (adjust for your GPU)
CLOCK_LOCKED=false
for FREQ in 1530 1410 1350 1200; do  # 1530 MHz is default for A6000
    if nvidia-smi -lgc $FREQ,$FREQ 2>/dev/null; then
        log_success "GPU clocks locked to $FREQ MHz"
        CLOCK_LOCKED=true
        break
    fi
done

if [ "$CLOCK_LOCKED" = false ]; then
    log_warn "Could not lock GPU clocks (may not be supported)"
    log_warn "Results may have higher variance"
fi

echo ""
log_info "GPU Status:"
nvidia-smi --query-gpu=name,clocks.gr,clocks.mem,power.draw --format=csv,noheader

# =============================================================================
# Main Benchmarks
# =============================================================================
if [ "$SKIP_BENCHMARK" = false ]; then

    # -------------------------------------------------------------------------
    # Latency vs T Benchmark
    # -------------------------------------------------------------------------
    log_section "Latency vs T Benchmark"
    log_info "Measuring generation latency across different sequence lengths"
    echo ""
    run_as_user python benchmark/main.py latency --version $VERSIONS --config "$CONFIG" --clear-log

    # -------------------------------------------------------------------------
    # VRAM vs T Benchmark
    # -------------------------------------------------------------------------
    log_section "VRAM vs T Benchmark"
    log_info "Measuring peak memory usage across different sequence lengths"
    echo ""
    run_as_user python benchmark/main.py vram --version $VERSIONS --config "$CONFIG"

    # -------------------------------------------------------------------------
    # Per-Phase Timing Benchmark
    # -------------------------------------------------------------------------
    log_section "Per-Phase Timing Benchmark"
    log_info "Measuring time spent in each model phase (embedding, attention, MLP, head)"
    echo ""
    run_as_user python benchmark/main.py phase --version $VERSIONS --config "$CONFIG"

    log_success "Main benchmarks complete!"

else
    log_section "Skipping Benchmarks"
    log_info "--skip-benchmark specified, using existing results"
fi

# =============================================================================
# Perplexity Evaluation
# =============================================================================
if [ "$SKIP_PERPLEXITY" = false ]; then
    log_section "Perplexity Evaluation"
    log_info "Evaluating model quality on WikiText-2 (autoregressive mode)"
    log_info "This tests that optimizations don't degrade output quality"
    echo ""

    # Run perplexity for all versions (autoregressive mode to test cache)
    run_as_user python benchmark/perplexity.py --autoregressive --version $VERSIONS

    log_success "Perplexity evaluation complete!"
else
    log_info "Skipping perplexity evaluation (--skip-perplexity)"
fi

# =============================================================================
# Roofline Profiling (optional, requires Nsight Compute)
# =============================================================================
if [ "$RUN_ROOFLINE" = true ]; then
    log_section "Roofline Profiling (Nsight Compute)"

    if ! command -v ncu &> /dev/null; then
        log_warn "Nsight Compute (ncu) not found in PATH"
        log_warn "Skipping roofline profiling"
    else
        log_info "Running Nsight Compute roofline analysis..."
        log_info "This profiles GPU compute vs memory bottlenecks"
        echo ""

        # Run roofline script
        bash benchmark/roofline/run_ncu.sh

        log_success "Roofline profiling complete!"
        log_info "Reports saved to benchmark/roofline/ncu_reports/"
    fi
fi

# =============================================================================
# Generate Plots
# =============================================================================
log_section "Generating Plots"
log_info "Creating visualizations from benchmark results"
echo ""

run_as_user python benchmark/plot_results.py \
    --results benchmark/results.jsonl \
    --output-dir benchmark/plots

log_success "All plots generated!"

# =============================================================================
# Summary
# =============================================================================
log_section "Summary"

echo "Results file:  benchmark/results.jsonl"
echo "Plots folder:  benchmark/plots/"
echo ""

# Count results
if [ -f benchmark/results.jsonl ]; then
    LATENCY_COUNT=$(grep -c '"benchmark_name": "latency_vs_T"' benchmark/results.jsonl 2>/dev/null || echo 0)
    VRAM_COUNT=$(grep -c '"benchmark_name": "vram_vs_T"' benchmark/results.jsonl 2>/dev/null || echo 0)
    PHASE_COUNT=$(grep -c '"benchmark_name": "per_phase_timing"' benchmark/results.jsonl 2>/dev/null || echo 0)
    PPL_COUNT=$(grep -c '"benchmark_name": "perplexity"' benchmark/results.jsonl 2>/dev/null || echo 0)

    echo "Benchmark records:"
    echo "  - latency_vs_T:     $LATENCY_COUNT"
    echo "  - vram_vs_T:        $VRAM_COUNT"
    echo "  - per_phase_timing: $PHASE_COUNT"
    echo "  - perplexity:       $PPL_COUNT"
    echo ""
fi

# List generated plots
echo "Generated plots:"
ls benchmark/plots/*.png 2>/dev/null | while read f; do
    echo "  - $(basename $f)"
done
echo ""

# Show metrics summary location
if [ -f "benchmark/plots/metrics_summary.json" ]; then
    echo "Metrics summary: benchmark/plots/metrics_summary.json"
fi

echo ""
log_success "Benchmark suite complete!"
echo ""
