#!/bin/bash
# B3: 定论 "draft 前向一轮几次" (HANDOFF 待办 2a / AUDIT F5)
# 手法: 固定一切, 只把 --spec-draft-n-max 从 1 -> 3 -> 7 变化, 看 draft_decode 是否随 n_max 线性.
#   线性  -> 每 token 一次前向 (n_max=7 时 8 次/轮) => draft 已带宽受限, 头寸 ~1.5x
#   不变  -> 一轮一次块前向                      => dispatch 受限, 头寸 8-12x
# NODROP=1: 诊断跑, 不对外报数字; 只看 draft_decode 的比值.
set -u
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
for N in 1 3 7; do
  echo "=========== ARM n_max=$N ==========="
  SPEC="--model-draft $D --spec-type draft-dflash --spec-draft-n-max $N" \
  CARDS=0,1,2 SPLIT=tensor TAG=b3-n$N NPRED=384 L=/root/libdir-nccl P2P=1 NODROP=1 PORT=812$N \
  bash /root/p60-ab-harness.sh
  echo "--- arm n_max=$N summary ---"
  tail -n 12 /tmp/p60-b3-n$N.log
  echo "--- arms so far: draft timing lines ---"
  grep -h -a -e "spec timing" /tmp/p60-b3-n*-server.log | tail -n 6
done
echo "B3_ALL_DONE"
