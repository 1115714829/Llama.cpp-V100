#!/bin/bash
set -uo pipefail
echo "=== /root/llm/models (all models) ==="
ls -la /root/llm/models/ 2>/dev/null
echo "=== /root/llm/models/Qwen3.8-27B-FP8/ ==="
ls -la /root/llm/models/Qwen3.8-27B-FP8/ 2>/dev/null | head -20
echo "=== /root/llm/systemd (services) ==="
ls -la /root/llm/systemd/ 2>/dev/null
echo "=== FP8 references in /root (scripts/logs) ==="
grep -rl "FP8" /root/*.sh /root/*.py /root/*.md /root/*.txt 2>/dev/null | head -15
echo "=== FP8 in /root/llm (configs/scripts) ==="
grep -rl "FP8" /root/llm/*.sh /root/llm/*.py /root/llm/*.md 2>/dev/null | head -10
echo "=== systemd units mentioning FP8 ==="
grep -l "FP8" /etc/systemd/system/*.service 2>/dev/null
echo FP8CHECK_DONE
