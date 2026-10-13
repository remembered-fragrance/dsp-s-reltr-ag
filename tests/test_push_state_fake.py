"""Kiểm push_state.py bằng kaggle CLI GIẢ (DSP_KAGGLE_CLI) — không chạm tài khoản Kaggle thật.
Chạy: python tests/test_push_state_fake.py"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAKE = r'''
import json, os, sys
from pathlib import Path
d = Path(os.environ["FAKE_DIR"]); a = sys.argv[1:]
with open(d / "calls.log", "a") as f: f.write(" ".join(a[:2]) + "\n")
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "rc0_forbidden":          # mô phỏng CLI 2.x: bị từ chối nhưng vẫn trả mã 0
    print("403 Client Error: Forbidden for url: https://api.kaggle.com/..."); sys.exit(0)
if a[:2] == ["datasets", "status"]:
    sys.exit(0 if (d / "exists").exists() else 1)
if a[:2] == ["datasets", "create"]:
    meta = json.load(open(Path(a[a.index("-p") + 1]) / "dataset-metadata.json"))
    assert meta["id"] == "nguyntaif/dsp-a1-state"
    (d / "exists").write_text("1"); print("Your private Dataset is being created."); sys.exit(0)
if a[:2] == ["datasets", "version"]:
    if mode == "progress":           # thanh tiến trình đi qua 401/403/404 MB nhưng upload thành công
        sys.stderr.write("\r 64%|######4   | 403M/623M [00:31<00:17, 12.9MB/s]\r100%|##########| 623M/623M [00:48<00:00, 13.4MB/s]\n")
        print("Upload successful: runs.zip (623MB)")
    print("Dataset version is being created."); sys.exit(0 if (d / "exists").exists() else 1)
sys.exit(2)
'''


def main():
    tmp = Path(tempfile.mkdtemp(prefix="push_fake_"))
    (tmp / "fake_kaggle.py").write_text(FAKE, encoding="utf-8")
    run = tmp / "runs" / "A1"
    (run / "ckpt").mkdir(parents=True)
    (run / "STATUS.json").write_text(json.dumps({"state": "running", "global_step": 10}))
    (run / "ckpt" / "last.pth").write_bytes(b"x" * 1000)
    (run / "ckpt" / "epoch1_weights.pth").write_bytes(b"w" * 100)
    env = dict(os.environ, DSP_KAGGLE_CLI=str(tmp / "fake_kaggle.py"), FAKE_DIR=str(tmp),
               KAGGLE_USERNAME="nguyntaif", KAGGLE_API_TOKEN="khong-phai-token-that", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, str(ROOT / "tools" / "push_state.py"), "--run", str(run), "--push_dir", str(tmp / "push")]
    r1 = subprocess.run(cmd + ["--msg", "lan 1"], env=env, capture_output=True, text=True)
    r2 = subprocess.run(cmd + ["--msg", "lan 2", "--with_weights"], env=env, capture_output=True, text=True)
    weights_pushed = (tmp / "push" / "runs" / "A1" / "ckpt" / "epoch1_weights.pth").exists()  # lần 3 sẽ dọn thư mục đẩy
    r3 = subprocess.run(cmd + ["--msg", "lan 3"], env=dict(env, FAKE_MODE="rc0_forbidden"), capture_output=True, text=True)
    r4 = subprocess.run(cmd + ["--msg", "lan 4"], env=dict(env, FAKE_MODE="progress"), capture_output=True, text=True)
    calls = (tmp / "calls.log").read_text().split("\n")
    print(r1.stdout.strip(), r2.stdout.strip(), r3.stdout.strip(), r4.stdout.strip(), sep="\n")
    checks = {
        "lần 1: dataset chưa có → create": r1.returncode == 0 and "create ok" in r1.stdout,
        "lần 2: đã có → version, kèm trọng số": r2.returncode == 0 and "version ok" in r2.stdout and weights_pushed,
        "lần 3: mã 0 nhưng 403 → báo LỖI": r3.returncode == 1 and "LỖI" in r3.stdout,
        "lần 4: thanh tiến trình có '403M' → vẫn ok": r4.returncode == 0 and "version ok" in r4.stdout,
        "chỉ CLI giả được gọi": calls[:4] == ["datasets status", "datasets create", "datasets status", "datasets version"],
    }
    for k, v in checks.items():
        print(("ok  " if v else "SAI ") + k)
    print("push_state (CLI giả):", "QUA" if all(checks.values()) else "KHÔNG QUA")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
