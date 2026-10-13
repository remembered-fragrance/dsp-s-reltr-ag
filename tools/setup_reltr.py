"""Dựng third_party/RelTR: clone RelTR upstream ở commit cố định rồi áp bản vá P1–P13 (patches/reltr_dsp.patch).

Repo không chứa mã RelTR (upstream không có LICENSE); CI, Space và máy mới đều dựng lại bằng script này.
  python tools/setup_reltr.py [--dst third_party/RelTR]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/yrcong/RelTR.git"
COMMIT = "fca7397e9aaeccd95541e83afa4b971f3fa89014"


def git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=str(ROOT / "third_party" / "RelTR"))
    ap.add_argument("--patch", default=str(ROOT / "patches" / "reltr_dsp.patch"))
    a = ap.parse_args()
    dst = Path(a.dst)
    if (dst / "UPSTREAM_COMMIT").exists():
        print(f"{dst} đã có (UPSTREAM_COMMIT), bỏ qua")
        return 0
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"{dst} đã tồn tại nhưng chưa vá: xóa thư mục này rồi chạy lại")
    dst.parent.mkdir(parents=True, exist_ok=True)
    git("clone", "--quiet", URL, str(dst))
    git("-c", "advice.detachedHead=false", "checkout", "--quiet", COMMIT, cwd=dst)
    git("apply", "--whitespace=nowarn", str(Path(a.patch).resolve()), cwd=dst)
    got = (dst / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip()
    assert got == COMMIT, f"UPSTREAM_COMMIT {got} != {COMMIT}"
    print(f"RelTR {COMMIT[:7]} + reltr_dsp.patch -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
