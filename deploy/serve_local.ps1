# Chạy demo trên máy nhà (GPU) và mở cho máy khác truy cập qua Cloudflare Quick Tunnel (miễn phí, không cần tài khoản).
#   powershell -ExecutionPolicy Bypass -File deploy/serve_local.ps1            # kéo code mới + demo + tunnel
#   powershell -ExecutionPolicy Bypass -File deploy/serve_local.ps1 -Share     # dự phòng: link *.gradio.live (hết hạn)
#   powershell -ExecutionPolicy Bypass -File deploy/serve_local.ps1 -NoPull    # không kéo code từ GitHub
# Cần: .venv có torch cu128; runs/A1/ckpt/epoch4_weights.pth; cloudflared (winget install --id Cloudflare.cloudflared).
# Link https://<ngẫu nhiên>.trycloudflare.com đổi mỗi lần chạy; ai có link phải nhập mật khẩu mới dùng được.
# Ctrl+C để dừng (tắt cả demo lẫn tunnel).
param([switch]$Share, [switch]$NoPull, [int]$Port = 7860, [string]$User = "dsp")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

if (-not $NoPull) {
    # CD tại chỗ: lấy code mới nhất của main (chỉ fast-forward, không đè thay đổi cục bộ)
    git pull --ff-only origin main
}

if (-not $env:DSP_DEMO_PASS) {
    $sec = Read-Host "Đặt mật khẩu cho demo (user: $User)" -AsSecureString
    $env:DSP_DEMO_PASS = [System.Net.NetworkCredential]::new('', $sec).Password
}
if ($env:DSP_DEMO_PASS.Length -lt 8) { throw "Mật khẩu cần ít nhất 8 ký tự" }
$env:DSP_DEMO_USER = $User
$env:DSP_PUBLIC = "1"
$env:PYTHONIOENCODING = "utf-8"

if ($Share) {
    & $Py demo/app.py --port $Port --share
    exit $LASTEXITCODE
}

$cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cf) { throw "Chưa có cloudflared: chạy  winget install --id Cloudflare.cloudflared  rồi mở terminal mới" }

$app = Start-Process -FilePath $Py -ArgumentList "demo/app.py", "--port", $Port -PassThru -NoNewWindow
try {
    Write-Host "Đang nạp model và khởi động demo..."
    $ok = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Seconds 2
        if ($app.HasExited) { throw "demo dừng với mã $($app.ExitCode)" }
        try { Invoke-WebRequest "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 2 | Out-Null; $ok = $true; break } catch {}
    }
    if (-not $ok) { throw "demo không phản hồi ở cổng $Port" }
    Write-Host "Demo chạy ở http://127.0.0.1:$Port. Mở tunnel; tìm dòng có 'trycloudflare.com' bên dưới để lấy link." -ForegroundColor Green
    & $cf tunnel --no-autoupdate --url "http://127.0.0.1:$Port"
}
finally {
    if (-not $app.HasExited) { Stop-Process -Id $app.Id -Force }
}
