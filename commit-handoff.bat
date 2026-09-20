@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
git add -A
git commit -m "archive : HANDOFF doc, HMMA/WMMA probes, card sweep, prefill measurements" -m "Adds the Volta HMMA/WMMA microbenchmark set (hmma_layout, hmma_map, hmma_probe, wmma_perf1-3, wmma_q8_bench), the card-count sweep (cards2) and prefill measurements, the ncu launch/poll tooling, the 1cat runtime-env dumps, and HANDOFF.md = the single authoritative hand-off for the V100/SM70 effort (current state, three landed wins, exhausted-lever list, next steps, service state, copy-paste section)." -m "Assisted-by: Qwen Code"
git log --oneline -2
echo --- status ---
git status --porcelain
echo DONE
