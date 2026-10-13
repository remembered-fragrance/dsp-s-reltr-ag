#!/usr/bin/env bash
# Linux/macOS: chạy demo và mở cho máy khác truy cập qua Cloudflare Quick Tunnel (miễn phí, có mật khẩu).
#   bash deploy/serve_local.sh            # kéo code mới + demo + tunnel
#   bash deploy/serve_local.sh --share    # dự phòng: link *.gradio.live tạm thời
#   bash deploy/serve_local.sh --no-pull  # không kéo code từ GitHub
# Cần: đã chạy python deploy/setup_demo.py; cloudflared (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PORT=${PORT:-7860}
SHARE=0; PULL=1
for arg in "$@"; do
  case "$arg" in
    --share) SHARE=1 ;;
    --no-pull) PULL=0 ;;
  esac
done
[ "$PULL" = 1 ] && git pull --ff-only origin main
if [ -z "${DSP_DEMO_PASS:-}" ]; then
  read -r -s -p "Đặt mật khẩu cho demo (user: ${DSP_DEMO_USER:-dsp}): " DSP_DEMO_PASS; echo
fi
[ "${#DSP_DEMO_PASS}" -ge 8 ] || { echo "Mật khẩu cần ít nhất 8 ký tự"; exit 1; }
export DSP_DEMO_PASS DSP_DEMO_USER="${DSP_DEMO_USER:-dsp}" DSP_PUBLIC=1
if [ "$SHARE" = 1 ]; then exec "$PY" demo/app.py --port "$PORT" --share; fi
command -v cloudflared >/dev/null || { echo "Chưa có cloudflared"; exit 1; }
"$PY" demo/app.py --port "$PORT" &
APP=$!
trap 'kill $APP 2>/dev/null' EXIT
echo "Đang nạp model và khởi động demo..."
for _ in $(seq 120); do
  sleep 2
  kill -0 $APP 2>/dev/null || { echo "demo đã dừng"; exit 1; }
  curl -fs "http://127.0.0.1:$PORT/" >/dev/null && break
done
echo "Demo chạy ở http://127.0.0.1:$PORT. Mở tunnel; tìm dòng có 'trycloudflare.com' để lấy link."
cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT"
