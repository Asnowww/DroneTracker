# Fetch the Seraphim archives straight from a mirror with curl.
#
# Why not huggingface_hub: on a machine behind a local proxy, the hub client's
# HEAD request fails even when the mirror is reachable, and its Xet transport
# opens hundreds of connections that the proxy accepts but never feeds. curl with
# --noproxy bypasses both problems and resumes cleanly with -C -.
#
#   powershell -File scripts/download_seraphim_mirror.ps1
#   powershell -File scripts/download_seraphim_mirror.ps1 -AllTrainBatches
#
# Two of the four train batches (~37k images) are plenty for a generic drone
# prior; pass -AllTrainBatches for the full 83k set.

param(
    [string]$Dest = "E:\airsim_yolov8_drone_tracker\datasets\seraphim",
    [string]$Endpoint = "https://hf-mirror.com",
    [switch]$AllTrainBatches
)

$repo = "datasets/lgrzybowski/seraphim-drone-detection-dataset"
$files = @(
    "train/labels/batch_001.zip",
    "test/labels/batch_001.zip",
    "test/images/batch_001.zip",
    "train/images/batch_001.zip",
    "train/images/batch_002.zip"
)
if ($AllTrainBatches) {
    $files += @("train/images/batch_003.zip", "train/images/batch_004.zip")
}

foreach ($rel in $files) {
    $out = Join-Path $Dest ($rel -replace '/', '\')
    $dir = Split-Path $out -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }

    $url = "$Endpoint/$repo/resolve/main/$rel"
    $before = if (Test-Path $out) { (Get-Item $out).Length } else { 0 }
    Write-Output "==> $rel (have $([math]::Round($before/1MB,1)) MB)"

    # -C - resumes; --retry-all-errors covers the abrupt TLS closes this mirror
    # sometimes throws mid-transfer.
    curl.exe --noproxy "*" -L -C - --retry 30 --retry-delay 5 --retry-all-errors `
        --connect-timeout 30 -o $out $url

    if (Test-Path $out) {
        Write-Output "    now $([math]::Round((Get-Item $out).Length/1MB,1)) MB"
    } else {
        Write-Output "    FAILED: $rel"
    }
}

Write-Output "--- totals ---"
Get-ChildItem $Dest -Recurse -File -Filter *.zip | ForEach-Object {
    "{0,10:N1} MB  {1}" -f ($_.Length / 1MB), $_.FullName.Substring($Dest.Length + 1)
}
