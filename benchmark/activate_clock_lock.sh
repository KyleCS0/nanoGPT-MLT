#!/bin/bash
# Lock GPU clocks for stable benchmarking
# For A6000: 1530 MHz (power-limited sustained clock)
# Adjust the frequency for your GPU

sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1530,1530
echo "GPU clocks locked to 1530 MHz"
nvidia-smi --query-gpu=clocks.gr --format=csv,noheader
