#!/bin/bash
# Is our draft a DFlash2 (with selector) or a plain DFlash v1? Inspect the GGUF metadata keys directly.
set -uo pipefail
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
echo "file: $D  size=$(stat -c %s "$D")"
echo ""
echo "=== dflash.* metadata keys (present = count>0) ==="
for k in dflash.block_size dflash.conv_kernel_size dflash.conv_group_size dflash.selector_rank dflash.selector_top_k dflash.sample_from_anchor dflash.attention.causal dflash.has_confidence_head; do
  printf "%-32s %s\n" "$k" "$(grep -a -c -- "$k" "$D" 2>/dev/null)"
done
echo ""
echo "=== selector / dspark tensors (present = count>0) ==="
for k in selector_hidden selector_prev selector_next markov_w1 markov_w2 conf_proj dspark; do
  printf "%-32s %s\n" "$k" "$(grep -a -c -- "$k" "$D" 2>/dev/null)"
done
echo ""
echo "=== architecture / general keys seen in the header ==="
grep -a -o -E "general\.architecture|dflash[a-z._]*|qwen[a-z0-9._]*|DSpark|DFlash2" "$D" 2>/dev/null | sort | uniq -c | sort -rn | head -20
echo ""
echo "=== 任何含 'selector' 的可读串 ==="
grep -a -o -E "[A-Za-z0-9_.]*selector[A-Za-z0-9_.]*" "$D" 2>/dev/null | sort | uniq -c | head -10
echo GGUF_INSPECT_DONE
