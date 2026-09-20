@echo off
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "sleep 90; grep -aE '^##### ARM|MEDIAN_TG=|^prompt[123]: |allreduce init|NCCL|libggml-cuda.so.0.24.0|health ok|P60_DONE|NCCL_AB_DONE' /tmp/nccl-ab-summary.log"
