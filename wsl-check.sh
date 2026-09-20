#!/bin/bash
# Check the WSL x86_64 toolchain for the llama.cpp tree (prerequisite for any code change).
echo "=== toolchain ==="
cmake --version 2>&1 | head -1
gcc --version 2>&1 | head -1
g++ --version 2>&1 | head -1
echo "nproc: $(nproc)"
echo ""
echo "=== tree ==="
T=/mnt/f/vllm+llama.cpp/llama.cpp
ls -d "$T" 2>/dev/null && echo "tree OK" || echo "TREE MISSING"
echo "existing build dirs:"
ls -d "$T"/build* 2>/dev/null || echo "  (none)"
echo ""
echo "=== writable? ==="
touch "$T/.wsl_write_test" 2>/dev/null && echo "writable" && rm -f "$T/.wsl_write_test" || echo "NOT writable"
echo ""
echo "=== ninja / make available? ==="
which ninja make 2>&1
echo WSL_CHECK_DONE
