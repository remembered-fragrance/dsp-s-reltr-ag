"""FR-1: trích frame AG từ video Charades 480p (require 6.5).

Giữ đúng ngữ nghĩa chỉ số frame của AG: `ffmpeg -loglevel panic -i <video> -q:v 2 <tmp>/%06d.jpg`
(đánh số từ 1, theo fps gốc, như tools/dump_frames.py của ActionGenome), rồi chỉ giữ frame cần dùng.
Thêm `-frames:v <chỉ số lớn nhất cần>`: chỉ cắt phần đuôi, không đổi cách đánh số.

Ví dụ (máy nhà):
  python tools/extract_frames.py --video_zip data/charades/Charades_v1_480.zip --ann_dir data/ag/annotations \
      --out_dir data/frames --splits test dev --workers 12
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_common as C  # noqa: E402

_ZIP = None
FAIL_FAST_N = 20


def find_ffmpeg(explicit=None):
    if explicit:
        return explicit
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("Không tìm thấy ffmpeg: cài imageio-ffmpeg hoặc truyền --ffmpeg")


def keep_frames(ann_dir, splits, k_max):
    """dict video -> list frame key (.png của AG) cần giữ, và split của từng video."""
    person, obj = C.load_ag(ann_dir, "full")
    train = C.sttran_filter(person, obj, "train")
    test = C.sttran_filter(person, obj, "test")
    C.check_ref_counts("full", "train", train, obj)
    C.check_ref_counts("full", "test", test, obj)
    dev = set(C.split_dev(train))
    need, split_of = {}, {}
    if "test" in splits:
        for v, fs in test.items():
            need[v], split_of[v] = list(fs), "test"
    for v, fs in train.items():
        if v in dev and "dev" in splits:
            need[v], split_of[v] = list(fs), "dev"
        elif v not in dev and "train" in splits:
            need[v], split_of[v] = C.even_frames(fs, k_max), "train"
    return need, split_of


def _jpeg_ok(path):
    """Đầu FF D8, cuối FF D9 — bắt file ghi dở hoặc toàn byte 0."""
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return False
            f.seek(-2, os.SEEK_END)
            return f.read(2) == b"\xff\xd9"
    except OSError:
        return False


def _open_zip(path):
    global _ZIP
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(path)
    return _ZIP


def _member_name(zf, video):
    for prefix in ("Charades_v1_480/", ""):
        if prefix + video in zf.NameToInfo:
            return prefix + video
    raise KeyError(f"không có {video} trong zip")


def extract_video(video, frames, out_dir, ffmpeg, video_zip=None, video_dir=None, tmp_root=None):
    """Trả về dict kết quả; chạy trong tiến trình con."""
    t0 = time.time()
    out_dir = Path(out_dir)
    dst_dir = out_dir / video
    targets = {C.frame_index(f): dst_dir / Path(C.jpg_name(f)).name for f in frames}
    if all(p.exists() for p in targets.values()) and (out_dir / "_done" / f"{video}.done").exists():
        return {"video": video, "need": len(targets), "ok": len(targets), "sec": 0.0, "status": "skip", "error": ""}
    tmp = Path(tempfile.mkdtemp(prefix="ag_", dir=tmp_root))
    try:
        if video_zip:
            zf = _open_zip(video_zip)
            src = tmp / video
            with zf.open(_member_name(zf, video)) as fin, open(src, "wb") as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)
        else:
            src = Path(video_dir) / video
        dump = tmp / "frames"
        dump.mkdir()
        # tùy chọn đầu ra phải đứng TRƯỚC đường dẫn đầu ra (ffmpeg bỏ qua tùy chọn đứng sau)
        cmd = [ffmpeg, "-loglevel", "panic", "-threads", "1", "-i", str(src),
               "-frames:v", str(max(targets)), "-q:v", "2", str(dump / "%06d.jpg")]
        rc = subprocess.run(cmd, stdin=subprocess.DEVNULL).returncode
        if rc != 0:  # ffmpeg bị ngắt/lỗi: không nhận frame nào (có thể đang ghi dở)
            return {"video": video, "need": len(targets), "ok": 0, "sec": round(time.time() - t0, 2), "status": "error",
                    "error": f"ffmpeg rc={rc}"}
        dst_dir.mkdir(parents=True, exist_ok=True)
        ok, missing = 0, []
        for idx, dst in targets.items():
            s = dump / f"{idx:06d}.jpg"
            if s.exists() and _jpeg_ok(s):
                # shutil.move: vùng tạm và thư mục đích có thể ở hai ổ khác nhau (Kaggle: /kaggle/tmp vs /kaggle/working),
                # os.replace sẽ lỗi EXDEV "Invalid cross-device link"
                shutil.move(str(s), str(dst))
                ok += 1
            else:
                missing.append(idx)
        status = "ok" if ok == len(targets) and all(_jpeg_ok(p) for p in targets.values()) else "missing"
        if status == "ok":
            for p in targets.values():  # ép ghi xuống đĩa trước khi đánh dấu xong (tắt máy đột ngột để lại file toàn byte 0)
                with open(p, "rb+") as fh:
                    os.fsync(fh.fileno())
            (out_dir / "_done").mkdir(exist_ok=True)
            (out_dir / "_done" / f"{video}.done").write_text(str(ok))
        err = f"rc={rc}; missing={missing[:10]}" if missing else ("" if rc == 0 else f"rc={rc}")
        return {"video": video, "need": len(targets), "ok": ok, "sec": round(time.time() - t0, 2), "status": status, "error": err}
    except Exception as e:  # ghi lại, không làm chết cả lượt chạy
        return {"video": video, "need": len(targets), "ok": 0, "sec": round(time.time() - t0, 2), "status": "error", "error": repr(e)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_zip")
    src.add_argument("--video_dir")
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"], choices=["train", "dev", "test"])
    ap.add_argument("--k_max", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--dev_ratio", type=float, default=C.DEV_RATIO)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--deadline_unix", type=float, default=0)
    ap.add_argument("--ffmpeg")
    ap.add_argument("--tmp_dir", default=None, help="thư mục tạm (Kaggle: /kaggle/tmp)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N video đầu (thử nghiệm)")
    args = ap.parse_args()
    assert args.dev_ratio == C.DEV_RATIO and args.seed == C.SEED, "dev_ratio/seed cố định theo require 6.6"

    ffmpeg = find_ffmpeg(args.ffmpeg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    need, split_of = keep_frames(args.ann_dir, set(args.splits), args.k_max)
    videos = sorted(need)
    if args.limit:
        videos = videos[: args.limit]
    keep_list = sorted(f"{v}/{Path(C.jpg_name(f)).name}" for v in videos for f in need[v])
    (out_dir / "_keep_list.txt").write_text("\n".join(keep_list) + "\n", encoding="utf-8")
    print(f"{len(videos)} video, {len(keep_list)} frame cần giữ; splits={args.splits}; ffmpeg={ffmpeg}", flush=True)

    log_path = out_dir / "_extract_log.csv"
    new_log = not log_path.exists()
    t0, done, n_fail, stopped = time.time(), 0, 0, False
    with open(log_path, "a", newline="", encoding="utf-8") as flog, ProcessPoolExecutor(args.workers) as ex:
        w = csv.DictWriter(flog, fieldnames=["video", "split", "need", "ok", "sec", "status", "error"])
        if new_log:
            w.writeheader()
        pending, it = set(), iter(videos)
        while True:
            while len(pending) < args.workers * 2 and not stopped:
                if C.deadline_passed(args.deadline_unix):
                    stopped = True
                    print("Tới hạn giờ (--deadline_unix): không nhận video mới", flush=True)
                    break
                v = next(it, None)
                if v is None:
                    break
                pending.add(ex.submit(extract_video, v, need[v], str(out_dir), ffmpeg, args.video_zip,
                                      args.video_dir, args.tmp_dir))
            if not pending:
                break
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fu in fin:
                r = fu.result()
                r["split"] = split_of[r["video"]]
                w.writerow(r)
                done += 1
                n_fail += r["status"] in ("missing", "error")
                if done >= FAIL_FAST_N and n_fail == done:  # lỗi hệ thống (không phải lỗi từng video): dừng sớm
                    flog.flush()
                    for p in pending:
                        p.cancel()
                    sys.exit(f"{done} video đầu tiên đều lỗi, ví dụ: {r['error']} — dừng để không tốn thời gian")
                if done % 100 == 0 or r["status"] in ("missing", "error"):
                    el = time.time() - t0
                    print(f"[{done}/{len(videos)}] {el / 60:.1f} phút; lỗi {n_fail}; {r['video']} {r['status']} {r['error']}", flush=True)
                    flog.flush()

    missing = [p for p in keep_list if not (out_dir / p).exists()]
    status = {"time": C.now_str(), "splits": args.splits, "videos": len(videos), "frames": len(keep_list),
              "missing_frames": len(missing), "failed_videos": n_fail, "minutes": round((time.time() - t0) / 60, 1),
              "state": "done" if not missing else ("time_limit" if stopped else "incomplete"),
              "ffmpeg": ffmpeg, "ffmpeg_version": subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.split("\n")[0]}
    C.dump_json(status, out_dir / "_extract_status.json", indent=1)
    print(json.dumps(status, ensure_ascii=False))
    if missing:
        print("Ví dụ frame thiếu:", missing[:5])
    return 0 if not missing or stopped else 1


if __name__ == "__main__":
    sys.exit(main())
