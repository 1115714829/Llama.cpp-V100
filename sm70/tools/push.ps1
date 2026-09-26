# push.ps1 - upload every sm70/tools file (*.sh, *.py, *.txt) to /root/llm/test/sm70/tools/ as LF, UTF-8 without BOM.
# Usage: powershell -ExecutionPolicy Bypass -File F:\vllm+llama.cpp\sm70\tools\push.ps1
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
# staging stays inside the workspace: the executor's sandbox only writes there
$tmp = Join-Path (Split-Path -Parent $here) 'results\raw\push-tmp'
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory $tmp | Out-Null
$utf8 = New-Object Text.UTF8Encoding($false)
$names = @()
Get-ChildItem $here -File | Where-Object { $_.Extension -in '.sh', '.py', '.txt' } | ForEach-Object {
    $t = [IO.File]::ReadAllText($_.FullName) -replace "`r`n", "`n"
    [IO.File]::WriteAllText((Join-Path $tmp $_.Name), $t, $utf8)
    $names += (Join-Path $tmp $_.Name)
}
ssh -o BatchMode=yes root@192.168.50.235 'mkdir -p /root/llm/test/sm70/tools /root/llm/test/sm70/logs'
scp -o BatchMode=yes -q @names root@192.168.50.235:/root/llm/test/sm70/tools/
ssh -o BatchMode=yes root@192.168.50.235 'chmod +x /root/llm/test/sm70/tools/*.sh /root/llm/test/sm70/tools/*.py; for f in /root/llm/test/sm70/tools/*.sh; do bash -n $f || echo SYNTAX_ERROR $f; done; for f in /root/llm/test/sm70/tools/*.py; do python3 -m py_compile $f || echo SYNTAX_ERROR $f; done; ls -l /root/llm/test/sm70/tools; echo TERM_OK_push'
