"""require 9.6: đẩy trạng thái train lên dataset Kaggle `<user>/dsp-a1-state` (chạy nền, do main.py gọi).

Đọc KAGGLE_USERNAME + (KAGGLE_API_TOKEN hoặc KAGGLE_KEY) từ biến môi trường (notebook nạp từ Kaggle Secrets).
Không in, không ghi token.
Không có token: ghi cảnh báo rồi thoát (vẫn dựa vào output của commit).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

FILES = ["STATUS.json", "log.txt", "sessions.csv", "train_steps.jsonl", "args.json", "A1.yaml", "push_log.txt"]
ERROR_MARKERS = ("error", "forbidden", "unauthorized", "denied", "failed")
ERROR_CODES = re.compile(r"\b40[134]\b")  # mã HTTP đứng riêng; không khớp số byte trong thanh tiến trình (vd "403M/623M")


def _is_progress(line):
    return "%|" in line


def kaggle(*args):
    """Gọi kaggle CLI. DSP_KAGGLE_CLI=<script .py> chỉ dùng khi KIỂM THỬ (CLI giả, không chạm tài khoản thật).
    kaggle CLI 2.x có thể trả mã 0 cả khi bị từ chối, nên thành công = mã 0 VÀ không có dấu hiệu lỗi trong output
    (bỏ qua dòng thanh tiến trình: 3/4 lần đẩy của phiên 1 bị báo LỖI giả vì số MB đi qua 401/403/404)."""
    fake = os.environ.get("DSP_KAGGLE_CLI")
    cmd = ([sys.executable, fake] if fake else ["kaggle"]) + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    text = "\n".join(l for l in out.splitlines() if not _is_progress(l)).lower()
    ok = r.returncode == 0 and not any(m in text for m in ERROR_MARKERS) and not ERROR_CODES.search(text)
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--push_dir", default="/kaggle/tmp/state_push")
    ap.add_argument("--msg", default="state")
    ap.add_argument("--with_weights", action="store_true")
    ap.add_argument("--dataset", default="dsp-a1-state")
    args = ap.parse_args()
    t0 = time.time()
    user = os.environ.get("KAGGLE_USERNAME")
    has_token = bool(os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN"))  # kiểu cũ hoặc kiểu mới (CLI 2.x)
    if not user or not has_token:
        print(f"[push] {time.strftime('%H:%M:%S')} không có token Kaggle trong môi trường: bỏ qua (require 9.6)")
        return 0
    run, push = Path(args.run), Path(args.push_dir)
    dst = push / "runs" / run.name
    if push.exists():
        shutil.rmtree(push)
    (dst / "ckpt").mkdir(parents=True)
    for f in FILES:
        if (run / f).exists():
            shutil.copy2(run / f, dst / f)
    for f in run.glob("env_session*.txt"):
        shutil.copy2(f, dst / f.name)
    if (run / "checks").exists():
        shutil.copytree(run / "checks", dst / "checks")
    shutil.copy2(run / "ckpt" / "last.pth", dst / "ckpt" / "last.pth")
    if args.with_weights:
        for w in sorted((run / "ckpt").glob("epoch*_weights.pth")):
            shutil.copy2(w, dst / "ckpt" / w.name)
    meta = {"title": args.dataset, "id": f"{user}/{args.dataset}", "licenses": [{"name": "other"}]}
    (push / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    # dataset đã có thì ra phiên bản mới, chưa có thì tạo (private) — kiểm bằng `datasets status`, không đoán từ thông báo lỗi
    exists, _ = kaggle("datasets", "status", meta["id"])
    if exists:
        ok, out = kaggle("datasets", "version", "-p", str(push), "-m", args.msg, "--dir-mode", "zip")
    else:
        ok, out = kaggle("datasets", "create", "-p", str(push), "--dir-mode", "zip")
    lines = [l for l in out.splitlines() if l.strip() and not _is_progress(l)]  # bỏ thanh tiến trình, dòng trống
    print(f"[push] {time.strftime('%H:%M:%S')} {args.msg}: {'version' if exists else 'create'} {'ok' if ok else 'LỖI'} "
          f"sau {time.time() - t0:.0f} giây. {' | '.join(lines)[-400:]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
