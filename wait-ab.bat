@echo off
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "sleep 420; grep -aE '^TAG=|^prompt[123]: |^MEDIAN_TG=|allreduce init|NCCL init|NCCL not compiled|internal AllReduce|target decode|spec timing|^libs:|P60_DONE|NCCL_AB_DONE' /tmp/nccl-ab-summary.log"
