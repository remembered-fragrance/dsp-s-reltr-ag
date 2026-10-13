"""Đóng gói thư mục cho Hugging Face Space: chỉ những file demo cần khi chạy.

  python tools/setup_reltr.py                     # cần third_party/RelTR đã vá
  python space/build_space.py --out build/space
Không chép trọng số, dữ liệu hay video: Space tải trọng số từ repo model private lúc khởi động.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ["ag_common.py", "ag_eval.py", "ag_pred.py", "demo.py", "extract_frames.py", "predict_ag.py"]
RELTR_KEEP = ["main.py", "engine.py", "UPSTREAM_COMMIT", "models", "datasets", "util", "lib"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "space"))
    a = ap.parse_args()
    out = Path(a.out)
    reltr = ROOT / "third_party" / "RelTR"
    if not (reltr / "UPSTREAM_COMMIT").exists():
        sys.exit("thiếu third_party/RelTR đã vá: chạy python tools/setup_reltr.py trước")
    if out.exists():
        shutil.rmtree(out)
    (out / "tools").mkdir(parents=True)
    (out / "demo").mkdir()
    shutil.copy2(ROOT / "space" / "README.md", out / "README.md")
    shutil.copy2(ROOT / "requirements.txt", out / "requirements.txt")
    shutil.copy2(ROOT / "demo" / "app.py", out / "demo" / "app.py")
    for f in TOOLS:
        shutil.copy2(ROOT / "tools" / f, out / "tools" / f)
    ign = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.so", "*.pyx", "setup.py", "demo", "*.png", "*.jpg")
    for name in RELTR_KEEP:
        src = reltr / name
        dst = out / "third_party" / "RelTR" / name
        if src.is_dir():
            shutil.copytree(src, dst, ignore=ign)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    files = [p for p in out.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    big = [p for p in files if p.stat().st_size > 5 * 2 ** 20]
    assert not big, f"file quá lớn cho Space: {big}"
    print(f"Space: {len(files)} file, {size / 2 ** 20:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
