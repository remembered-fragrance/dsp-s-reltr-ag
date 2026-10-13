"""FR-11: chuẩn bị thư mục upload Kaggle và in lệnh cho người dùng chạy (không đọc kaggle.json).

  python tools/kaggle_stage.py --user <username> [--datasets code annotations ckpt] [--update "ghi chú"]

Tạo kaggle_upload/<dataset>/ (+ dataset-metadata.json, private). dsp-code kèm CODE_VERSION.json (thời điểm,
sha256 cây code, commit RelTR gốc) để log của mỗi lần train ghi lại phiên bản code đã dùng (MR-8).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_PARTS = ["third_party/RelTR", "tools", "configs", "notebooks", "tests", "demo"]
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIX = {".pyc", ".so", ".png"}


def iter_code_files():
    for part in CODE_PARTS:
        base = ROOT / part
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS) and p.suffix not in SKIP_SUFFIX:
                yield p


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_meta(d, user, name):
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": name, "id": f"{user}/{name}", "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--datasets", nargs="+", default=["code", "annotations", "ckpt"], choices=["code", "annotations", "ckpt", "data"])
    ap.add_argument("--out", default=str(ROOT / "kaggle_upload"))
    ap.add_argument("--update", default="", help="ghi chú phiên bản mới (dùng `kaggle datasets version` thay vì create)")
    args = ap.parse_args()
    out = Path(args.out)
    cmds = []
    exe = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    kaggle = str(exe.relative_to(ROOT)) if exe.exists() and ROOT in exe.parents else "kaggle"

    def cmd_for(d):
        rel = d.relative_to(ROOT) if ROOT in d.parents else d
        if args.update:
            return f'{kaggle} datasets version -p {rel} -m "{args.update}" --dir-mode zip'
        return f'{kaggle} datasets create -p {rel} --dir-mode zip'

    if "code" in args.datasets:
        d = out / "dsp-code"
        if d.exists():
            shutil.rmtree(d)
        h = hashlib.sha256()
        n = 0
        for p in iter_code_files():
            rel = p.relative_to(ROOT)
            h.update(str(rel.as_posix()).encode())
            h.update(p.read_bytes())
            dst = d / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            n += 1
        # bằng chứng các cổng chạy ở máy nhà mà notebook Kaggle cần đọc (G4 — require mục 10)
        for name in ("g4_overfit_local.json",):
            p = ROOT / "checks" / name
            if p.exists():
                (d / "checks").mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d / "checks" / name)
                h.update(p.read_bytes())
            else:
                print(f"CẢNH BÁO: thiếu checks/{name} — 01-checks sẽ dừng ở cổng G4")
        ver = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": n, "tree_sha256": h.hexdigest(),
               "reltr_upstream_commit": (ROOT / "third_party/RelTR/UPSTREAM_COMMIT").read_text().strip(),
               "note": args.update or "initial"}
        (d / "CODE_VERSION.json").write_text(json.dumps(ver, indent=1), encoding="utf-8")
        write_meta(d, args.user, "dsp-code")
        cmds.append(cmd_for(d))
        print(f"dsp-code: {n} file, tree_sha256 {ver['tree_sha256'][:16]}")
    if "annotations" in args.datasets:
        d = out / "dsp-ag-annotations"
        src = ROOT / "data" / "ag" / "annotations"
        names = ["object_bbox_and_relationship.pkl", "object_bbox_and_relationship_filtersmall.pkl", "person_bbox.pkl",
                 "frame_list.txt", "object_classes.txt", "relationship_classes.txt"]
        for nme in names:
            assert (src / nme).exists(), f"thiếu {src / nme}"
            link_or_copy(src / nme, d / nme)
        write_meta(d, args.user, "dsp-ag-annotations")
        cmds.append(cmd_for(d))
        print(f"dsp-ag-annotations: {len(names)} file")
    if "ckpt" in args.datasets:
        d = out / "reltr-vg-ckpt"
        link_or_copy(ROOT / "data" / "ckpt" / "checkpoint0149.pth", d / "checkpoint0149.pth")
        write_meta(d, args.user, "reltr-vg-ckpt")
        cmds.append(cmd_for(d))
        print("reltr-vg-ckpt: checkpoint0149.pth")
    if "data" in args.datasets:
        # dsp-ag-data trích ở máy nhà (require 6.1, phương án upload frame): data/{frames,reltr,reltr_overfit} + bằng chứng G1–G3
        d = out / "dsp-ag-data"
        if d.exists():
            shutil.rmtree(d)
        frames, reltr, overfit = ROOT / "data" / "frames", ROOT / "data" / "reltr", ROOT / "data" / "reltr_overfit"
        st = json.loads((frames / "_extract_status.json").read_text(encoding="utf-8"))
        assert st["state"] == "done" and set(st["splits"]) == {"train", "dev", "test"}, f"frame chưa trích đủ 3 split: {st}"
        meta = json.loads((reltr / "meta.json").read_text(encoding="utf-8"))
        assert set(meta["splits"]) == {"train", "dev", "test"} and meta["frames_checked"], "data/reltr chưa sinh đủ 3 split có kiểm frame"
        assert (overfit / "overfit_frames.txt").exists(), "thiếu data/reltr_overfit (convert_ag --overfit 64)"
        keep = [l.strip() for l in open(frames / "_keep_list.txt", encoding="utf-8") if l.strip()]
        for rel in keep:
            link_or_copy(frames / rel, d / "data" / "frames" / rel)
        for f in ("_keep_list.txt", "_extract_status.json", "_extract_log.csv"):
            link_or_copy(frames / f, d / "data" / "frames" / f)
        for src_dir in (reltr, overfit):
            for p in src_dir.iterdir():
                if p.is_file():
                    link_or_copy(p, d / "data" / src_dir.name / p.name)
        for p in [ROOT / "checks" / n for n in ("g1_frames_local.csv", "g2_stats.txt", "g2_oed_match.json",
                                                 "extract_frames_local_status.json")] + [ROOT / "checks" / "g3_gt_as_pred" / "metrics.json"]:
            if p.exists():
                link_or_copy(p, d / "checks" / p.relative_to(ROOT / "checks"))
        write_meta(d, args.user, "dsp-ag-data")
        cmds.append(cmd_for(d))
        print(f"dsp-ag-data: {len(keep)} frame, k={meta['k']}, ffmpeg: {st.get('ffmpeg_version', '?')[:40]}")
    print("\nChạy các lệnh sau (PowerShell, trong .venv đã cài kaggle; dataset mặc định là private):")
    for c in cmds:
        print("  " + c)


if __name__ == "__main__":
    sys.exit(main())
