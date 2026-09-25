# make-manifest.ps1 - regenerate the source manifest of the llama.cpp working tree.
#
#   powershell -File .tools/make-manifest.ps1
#
# Writes .tools/tree-md5-local.txt: CR-normalized md5 of every source file,
# LF line endings, sorted by md5. This file is uploaded to the build host as
# /tmp/tree-md5-local.txt and is the gate for /tmp/p4-build.sh.
$ErrorActionPreference = 'Stop'
$root  = 'F:\vllm+llama.cpp\llama.cpp'
$out   = 'F:\vllm+llama.cpp\.tools\tree-md5-local.txt'
$exts  = @('.c', '.cpp', '.cu', '.cuh', '.h', '.hpp')

$files = Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $exts -contains $_.Extension.ToLower() -and
    $_.FullName -notmatch '\\build' -and $_.FullName -notmatch '\\\.git\\'
}

$md5  = [Security.Cryptography.MD5]::Create()
$rows = New-Object System.Collections.Generic.List[string]
foreach ($f in $files) {
    $rel = $f.FullName.Substring($root.Length + 1).Replace('\', '/')
    $txt = [IO.File]::ReadAllText($f.FullName) -replace "`r`n", "`n"
    $hex = ($md5.ComputeHash([Text.Encoding]::UTF8.GetBytes($txt)) | ForEach-Object { $_.ToString('x2') }) -join ''
    $rows.Add("$hex  $rel")
}

$sorted = $rows | Sort-Object
[IO.File]::WriteAllText($out, ($sorted -join "`n") + "`n", (New-Object Text.UTF8Encoding($false)))
Write-Output "MANIFEST=$out FILES=$($rows.Count)"
