#!/bin/bash
# Build a fixed perplexity corpus on the server (deterministic, identical for both arms).
set -u
SRC=/root/llm/test/v100-opt/llama.cpp
OUT=/root/ppl-corpus.txt
cat "$SRC/README.md" "$SRC/docs"/*.md "$SRC/tools/server/README.md" "$SRC/AGENTS.md" > "$OUT" 2>/dev/null
echo "corpus bytes: $(wc -c < "$OUT")"
echo "corpus lines: $(wc -l < "$OUT")"
md5sum "$OUT"
echo CORPUS_DONE
