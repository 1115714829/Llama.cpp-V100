# dsh-run.ps1 - run one DSH headless task (the executor) and capture its output. Main agent only.
# Usage: powershell -ExecutionPolicy Bypass -File sm70\tools\dsh-run.ps1 -Name <log name> -Task "<task text>" [-Mode workspace-write|danger-full-access]
# stdout = the executor's final message, stderr = its streamed reasoning; both land in sm70\results\raw\dsh\.
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Task,
    [string]$Mode = 'workspace-write'
)
$node  = 'C:\Users\a1115\AppData\Local\pi-node\current\node.exe'
$entry = 'C:\Program Files\DSH Desktop\resources\harness-node-entry.mjs'
$bin   = 'C:\Program Files\DSH Desktop\resources\app\node_modules\@deepseek-ai\dsh\lib\bin.js'
$root  = 'F:\vllm+llama.cpp'
$logs  = Join-Path $root 'sm70\results\raw\dsh'
New-Item -ItemType Directory -Force $logs | Out-Null
$out = Join-Path $logs "$Name.out.txt"
$err = Join-Path $logs "$Name.err.txt"
$env:DSH_PERMISSION_MODE = $Mode
$argv = @("`"$entry`"", "`"$bin`"", '--profile', 'headless', ('"' + ($Task -replace '"', '\"') + '"'))
$t0 = Get-Date
$p = Start-Process -FilePath $node -ArgumentList $argv -WorkingDirectory $root `
    -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -Wait -PassThru
"DSH_RUN_DONE name=$Name rc=$($p.ExitCode) mode=$Mode seconds=$([int]((Get-Date) - $t0).TotalSeconds) out=$out err=$err"
