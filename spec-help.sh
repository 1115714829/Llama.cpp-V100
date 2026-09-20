#!/bin/bash
# List speculative-decoding knobs available in b11053.
set -uo pipefail
B=/root/libdir-final/llama-server
echo "=== all --spec-* options ==="
"$B" --help 2>&1 | grep -E "^--spec" | head -40
echo ""
echo "=== n-min / p-min / n-max related ==="
"$B" --help 2>&1 | grep -iE "n-min|n_min|p-min|p_min|n-max|n_max|draft" | head -30
echo HELP_DONE
