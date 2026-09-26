# tree-md5.ps1 - workstation twin of tree-md5.sh (identical output format and ordering).
# Usage: powershell -ExecutionPolicy Bypass -File sm70\tools\tree-md5.ps1 [-Tree <dir>] [-Out <file>]
param(
    [string]$Tree = 'F:\vllm+llama.cpp\llama.cpp',
    [string]$Out  = 'F:\vllm+llama.cpp\sm70\results\raw\manifest-local.txt'
)
$ErrorActionPreference = 'Stop'
$exts = @('.c', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.cmake')
$root = (Resolve-Path $Tree).Path.TrimEnd('\')
$md5 = [Security.Cryptography.MD5]::Create()
# ISO-8859-1 maps every byte to one char, so removing CR through it is byte-exact
$latin1 = [Text.Encoding]::GetEncoding(28591)
$cr = [string][char]13
$top = Get-ChildItem -LiteralPath $root -Force | Where-Object { -not ($_.PSIsContainer -and ($_.Name -like 'build*' -or $_.Name -eq '.git')) }
$files = foreach ($t in $top) {
    if ($t.PSIsContainer) { Get-ChildItem -LiteralPath $t.FullName -Recurse -File -Force } else { $t }
}
$entries = New-Object System.Collections.Generic.List[string]
foreach ($f in $files) {
    if (($exts -ccontains $f.Extension) -or ($f.Name -ceq 'CMakeLists.txt')) {
        $entries.Add($f.FullName.Substring($root.Length + 1).Replace('\', '/'))
    }
}
$arr = $entries.ToArray()
[Array]::Sort($arr, [StringComparer]::Ordinal)
$lines = foreach ($rel in $arr) {
    $text = $latin1.GetString([IO.File]::ReadAllBytes((Join-Path $root $rel))).Replace($cr, '')
    (($md5.ComputeHash($latin1.GetBytes($text)) | ForEach-Object { $_.ToString('x2') }) -join '') + '  ' + $rel
}
New-Item -ItemType Directory -Force (Split-Path -Parent $Out) | Out-Null
[IO.File]::WriteAllText($Out, (($lines -join "`n") + "`n"), (New-Object Text.UTF8Encoding($false)))
"TREE_MD5 files=$($arr.Length) out=$Out"
