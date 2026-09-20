@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
scp -o BatchMode=yes root@192.168.50.235:/tmp/ncu-tgt-server.log ./ncu-tgt-server.log
git add -A
git commit -m "archive : Goal 8dbd30ea step 1 -- GDN vs attention op costs; ncu is dead on AC922" -m "Measures the M=8 verify round's GDN and attention cost at op level (ncu cannot profile this workload: device-memory save/restore fails on the 29 GB Q8_0 model across 3 cards). Result: GDN ~16 us/layer and FA ~26 us/layer at short context, so 64 layers total only 1-2 ms of a ~30 ms target forward (3-6%). Redirects the plan to the draft forward (8-12x headroom, still unquantified) and to the ~21 ms unaccounted per round. Also records two new tooling traps: a script's 'rm -rf /tmp/X.*' deletes its own log, and pkill -f self-matches the remote shell. Adds the raw op-benchmark outputs and the ncu failure evidence log." -m "Assisted-by: Qwen Code"
git log --oneline -3
echo --- status ---
git status --porcelain
echo DONE
