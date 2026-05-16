param(
    [int]$LargeFileMegabytes = 20
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$largeLimit = $LargeFileMegabytes * 1MB

$blockedNames = @(
    "DataSet", "DES", "EmoPro", "Exp", "Latex", "Paper", "Pie_Chart", "Reference",
    "data", "dataset", "datasets", "images", "image", "annotations", "annotation",
    "masks", "mask", "checkpoints", "outputs", "runs"
)

$blockedExtensions = @(
    ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z", ".xlsx", ".xls", ".csv",
    ".pdf", ".docx", ".pptx", ".psd", ".ai", ".bmp", ".pt", ".pth", ".ckpt",
    ".safetensors", ".pkl"
)

$items = Get-ChildItem -LiteralPath $root -Force -Recurse |
    Where-Object {
        $_.FullName -notmatch "\\.git(\\|$)"
    }

$largeFiles = $items |
    Where-Object { -not $_.PSIsContainer -and $_.Length -ge $largeLimit } |
    Select-Object FullName, Length

$blockedFiles = $items |
    Where-Object { -not $_.PSIsContainer -and $blockedExtensions -contains $_.Extension.ToLowerInvariant() } |
    Select-Object FullName, Length

$blockedDirs = $items |
    Where-Object { $_.PSIsContainer -and $blockedNames -contains $_.Name } |
    Select-Object FullName

if ($largeFiles -or $blockedFiles -or $blockedDirs) {
    Write-Host "Release check found files or folders that should stay out of Git." -ForegroundColor Yellow

    if ($largeFiles) {
        Write-Host "`nLarge files:" -ForegroundColor Yellow
        $largeFiles | Format-Table -AutoSize
    }

    if ($blockedFiles) {
        Write-Host "`nBlocked file types:" -ForegroundColor Yellow
        $blockedFiles | Format-Table -AutoSize
    }

    if ($blockedDirs) {
        Write-Host "`nBlocked directories:" -ForegroundColor Yellow
        $blockedDirs | Format-Table -AutoSize
    }

    exit 1
}

Write-Host "Release check passed. No large local dataset artifacts were found." -ForegroundColor Green
