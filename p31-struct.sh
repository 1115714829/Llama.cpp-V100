#!/bin/bash
# Structural comparison: why does DFlash2 work on Q2_K_XL but collapse on TurboFCFusion Q4_K_M?
set -uo pipefail
for f in /tmp/iso27858-q2xl_tensor_dflash7.log /tmp/mg27858-tensor_dflash7.log; do
  echo "########## $f"
  echo "-- target meta (first occurrence = target, second = draft) --"
  grep -aE "n_layer +=|n_embd +=|n_vocab +=|n_head +=|n_head_kv +=" "$f" | head -12
  echo "-- dflash/draft config --"
  grep -aiE "target_layer_ids|n_extract|block_size|DFlash2 conv kernel|extract layer" "$f" | head -8
  echo ""
done
echo "########## chat-template / stop-format sanity (did the model actually answer?)"
for f in /tmp/iso27858-q2xl_tensor_dflash7.log /tmp/mg27858-tensor_dflash7.log; do
  echo "-- $f"
  grep -acE "stop_type" "$f" 2>/dev/null | head -1
done
echo STRUCT_DONE
