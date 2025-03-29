#!/bin/bash
NCU=/usr/local/NVIDIA-Nsight-Compute-2025.1/ncu
$NCU --version
sudo env PATH="$PATH" VIRTUAL_ENV="$VIRTUAL_ENV" $NCU -f --section SchedulerStats --section WarpStateStats --replay-mode kernel -o report.ncu-rep   --target-processes all   python basic.py