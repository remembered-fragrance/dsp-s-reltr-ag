"""FR-8: cell kết thúc của notebook 02 (chạy cả khi train lỗi).

Ghi một dòng vào sessions.csv, cập nhật STATUS.json nếu tiến trình train chết mà chưa kịp ghi (failed),
chờ lần đẩy trạng thái cuối (tối đa --wait_push_min phút), in hướng dẫn cho phiên tiếp theo.
"""
import argparse
import csv
import datetime
import os
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "third_party" / "RelTR"))
from util import checkpoint as ck  # noqa: E402


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--rc", default="None")
    ap.add_argument("--wait_push_min", type=float, default=15)
    args = ap.parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    rc = None if args.rc in ("None", "") else int(args.rc)
    st = ck.read_status(run)
    if rc not in (None, 0) and st.get("state") in (None, "running"):
        st = ck.update_status(run, state="failed", message=f"tiến trình train thoát với mã {rc}")

    pid_file = run / "push.pid"
    if pid_file.exists():
        pid, t = int(pid_file.read_text().strip()), time.time()
        while pid_alive(pid) and time.time() - t < args.wait_push_min * 60:
            time.sleep(10)
        print(f"Đẩy trạng thái: {'xong' if not pid_alive(pid) else 'CHƯA xong sau ' + str(args.wait_push_min) + ' phút'}")

    st = ck.read_status(run)
    row = {"session": st.get("session"), "t0": datetime.datetime.fromtimestamp(args.t0).isoformat(timespec="seconds"),
           "end": datetime.datetime.now().isoformat(timespec="seconds"),
           "hours": round((time.time() - args.t0) / 3600, 2), "start_step": st.get("session_start_step"),
           "end_step": st.get("global_step"), "rc": rc, "state": st.get("state"), "message": st.get("message", "")}
    f = run / "sessions.csv"
    new = not f.exists()
    with open(f, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print("sessions.csv:", row)
    nxt = {"done": "Train xong. Tải dsp-a1-state (hoặc output notebook) về máy nhà, rồi làm Bước 5 (G8, test).",
           "time_limit": "Dừng vì hết giờ. Gắn output phiên này (hoặc dsp-a1-state mới nhất) làm input, rồi commit lại 02-train.",
           "max_iters": "Dừng vì --max_iters (chỉ dùng khi kiểm tra).",
           "failed": "Train lỗi: xem log.txt, STATUS.json (message), đoạn cuối output của cell train."}
    print("Bước tiếp theo:", nxt.get(st.get("state"), f"trạng thái lạ: {st.get('state')}"))


if __name__ == "__main__":
    main()
