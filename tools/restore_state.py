"""FR-8 / require 9.5: khôi phục trạng thái train đầu notebook 02.

Tìm checkpoint hợp lệ mới nhất trong --search (các input Kaggle: dsp-a1-state, output phiên trước) và --dst,
chép toàn bộ thư mục run chứa nó sang --dst. Dòng cuối in JSON:
  {"mode": "fresh" | "resume" | "done", "ckpt": <đường dẫn hoặc null>, "global_step": <int>}
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "third_party" / "RelTR"))
from util import checkpoint as ck  # noqa: E402


def unpack_zips(search, tmp):
    """Dataset đẩy bằng --dir-mode zip có thể còn ở dạng .zip: giải nén các zip có chứa last.pth."""
    roots = []
    for z in Path(search).rglob("*.zip"):
        try:
            with zipfile.ZipFile(z) as zf:
                if any(n.endswith(("last.pth", "STATUS.json")) for n in zf.namelist()):
                    dst = Path(tmp) / z.stem
                    zf.extractall(dst)
                    roots.append(dst)
        except zipfile.BadZipFile:
            pass
    return roots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", nargs="+", default=["/kaggle/input"])
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tmp", default="/kaggle/tmp/restore")
    ap.add_argument("--run_name", default=None,
                    help="chỉ nhận checkpoint trong thư mục run tên này (02-train: A1), tránh resume nhầm từ run thử của G4–G6")
    args = ap.parse_args()
    dst = Path(args.dst)

    roots = [Path(s) for s in args.search] + [dst]
    for s in args.search:
        if Path(s).exists():
            roots += unpack_zips(s, args.tmp)
    ok_run = (lambda run_dir: True) if not args.run_name else (lambda run_dir: run_dir.name == args.run_name)
    path, _, gs = ck.select_checkpoint(roots, load=False, run_filter=ok_run)

    # trạng thái 'done' có thể nằm ở thư mục không còn last.pth hợp lệ: kiểm cả STATUS.json
    done_dirs = [p.parent for r in roots if r.exists() for p in r.rglob("STATUS.json")
                 if ok_run(p.parent) and ck.read_status(p.parent).get("state") == "done"]

    if path is None and not done_dirs:
        print("Không có checkpoint nào: train mới với --pretrained")
        print(json.dumps({"mode": "fresh", "ckpt": None, "global_step": 0}))
        return
    src_run = path.parent.parent if path is not None else done_dirs[0]
    if src_run.resolve() != dst.resolve():
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_run, dst, dirs_exist_ok=True)
        print(f"Đã chép {src_run} -> {dst}")
    st = ck.read_status(dst)
    if st.get("state") == "done":
        w = dst / "ckpt" / f"epoch{st.get('epochs', 4)}_weights.pth"
        print(f"STATUS.json = done; trọng số cuối: {w} (có: {w.exists()})")
        print(json.dumps({"mode": "done", "ckpt": str(w) if w.exists() else None, "global_step": st.get("global_step")}))
        return
    local = dst / "ckpt" / path.name
    print(json.dumps({"mode": "resume", "ckpt": str(local), "global_step": gs}))


if __name__ == "__main__":
    main()
