#!/bin/bash
echo -n 'orphan_loops: '; ps -eo args | grep -a -c 'n=0; while'
echo -n 'llama_server: '; pgrep -x llama-server | wc -l
echo -n 'llama_bench: '; pgrep -x llama-bench | wc -l
echo -n 'cc1plus: '; pgrep -x cc1plus | wc -l
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo VERIFY_DONE