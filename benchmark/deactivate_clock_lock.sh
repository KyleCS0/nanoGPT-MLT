#!/bin/bash
# Reset GPU clocks to default after benchmarking

sudo nvidia-smi -rgc
echo "GPU clocks reset to default"
nvidia-smi --query-gpu=clocks.gr --format=csv,noheader
