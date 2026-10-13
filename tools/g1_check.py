"""G1 (require mục 10): kiểm chỉ số frame.

Chọn N video đã trích (seed cố định), dump đủ frame dạng PNG bằng đúng lệnh của AG
(`ffmpeg -loglevel panic -i <video> <dir>/%06d.png`), rồi với mỗi frame JPEG đã giữ ở chỉ số i:
sai khác với PNG[i] phải nhỏ hơn sai khác với PNG[i-1] và PNG[i+1].
"""
import argparse
import csv
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_common as C  # noqa: E402
from extract_frames import _member_name, find_ffmpeg  # noqa: E402


def mad(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_zip", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out", required=True, help="checks/g1_frames.csv")
    ap.add_argument("--ffmpeg")
    args = ap.parse_args()

    ffmpeg = find_ffmpeg(args.ffmpeg)
    frames_dir = Path(args.frames_dir)
    done = sorted(p.stem for p in (frames_dir / "_done").glob("*.done"))
    assert done, "chưa có video nào trích xong"
    videos = random.Random(args.seed).sample(done, min(args.n, len(done)))
    zf = zipfile.ZipFile(args.video_zip)
    rows, n_bad = [], 0
    for v in videos:
        tmp = Path(tempfile.mkdtemp(prefix="g1_"))
        try:
            src = tmp / v
            with zf.open(_member_name(zf, v)) as fi, open(src, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
            (tmp / "png").mkdir()
            subprocess.run([ffmpeg, "-loglevel", "panic", "-i", str(src), str(tmp / "png" / "%06d.png")], check=True)
            for jpg in sorted((frames_dir / v).glob("*.jpg")):
                i = int(jpg.stem)
                ref = np.asarray(Image.open(jpg).convert("RGB"))
                d = {}
                for j in (i - 1, i, i + 1):
                    p = tmp / "png" / f"{j:06d}.png"
                    if p.exists():
                        d[j] = mad(ref, np.asarray(Image.open(p).convert("RGB")))
                ok = i in d and all(d[i] < d[j] for j in d if j != i)
                n_bad += not ok
                rows.append({"video": v, "frame": i, "mad_prev": d.get(i - 1), "mad_same": d.get(i),
                             "mad_next": d.get(i + 1), "pass": ok})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"G1: {len(rows)} frame của {len(videos)} video ({', '.join(videos)}); không đạt: {n_bad}")
    print("G1", "QUA" if n_bad == 0 else "KHÔNG QUA")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
