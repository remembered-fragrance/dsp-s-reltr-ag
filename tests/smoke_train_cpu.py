"""Smoke test CPU cho main.py (P1, P4, P9–P13) + restore_state, finalize_session, g6_check — trước khi tốn quota Kaggle.

Dùng nhãn AG thật (data/ag/annotations) và JSON từ convert_ag (--skip_frames_check), ảnh là nhiễu ngẫu nhiên
đúng kích thước. Nếu máy chưa có pycocotools, dùng một bản COCO tối giản (chỉ cho smoke test).
  python tests/smoke_train_cpu.py --work <thư mục tạm> --reltr_json <out_dir của convert_ag dry-run>
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RELTR = ROOT / "third_party" / "RelTR"
TOOLS = ROOT / "tools"

STUB_COCO = '''import json
from collections import defaultdict
class COCO:
    def __init__(self, annotation_file=None):
        self.dataset, self.anns, self.imgs, self.cats = {}, {}, {}, {}
        self.imgToAnns = defaultdict(list)
        if annotation_file:
            self.dataset = json.load(open(annotation_file, encoding="utf-8")); self.createIndex()
    def createIndex(self):
        for a in self.dataset.get("annotations", []):
            self.anns[a["id"]] = a; self.imgToAnns[a["image_id"]].append(a)
        for i in self.dataset.get("images", []): self.imgs[i["id"]] = i
        for c in self.dataset.get("categories", []): self.cats[c["id"]] = c
    def getAnnIds(self, imgIds=[], **kw):
        imgIds = imgIds if isinstance(imgIds, (list, tuple)) else [imgIds]
        return [a["id"] for i in imgIds for a in self.imgToAnns[i]]
    def loadAnns(self, ids): return [self.anns[i] for i in (ids if isinstance(ids, (list, tuple)) else [ids])]
    def loadImgs(self, ids): return [self.imgs[i] for i in (ids if isinstance(ids, (list, tuple)) else [ids])]
'''


def make_stub(d):
    p = d / "pycocotools"
    p.mkdir(parents=True, exist_ok=True)
    (p / "__init__.py").write_text("")
    (p / "coco.py").write_text(STUB_COCO)
    (p / "mask.py").write_text("")
    (p / "cocoeval.py").write_text("class COCOeval: pass\n")


def run(cmd, env, cwd=RELTR, check=True):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr)[-2500:]
    if check and r.returncode != 0:
        print(tail)
        raise SystemExit(f"lỗi mã {r.returncode}")
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--reltr_json", required=True)
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--n_dev", type=int, default=4)
    a = ap.parse_args()
    W = Path(a.work)
    shutil.rmtree(W, ignore_errors=True)
    W.mkdir(parents=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        import pycocotools  # noqa: F401
    except ImportError:
        make_stub(W / "stubs")
        env["PYTHONPATH"] = str(W / "stubs") + os.pathsep + env.get("PYTHONPATH", "")
        print("dùng COCO tối giản (máy chưa có pycocotools)")

    # ---- dữ liệu nhỏ: n_train frame train, val = dev đầy đủ, dev300 = n_dev frame đầu
    src = Path(a.reltr_json)
    tr = json.load(open(src / "train.json", encoding="utf-8"))
    val = json.load(open(src / "val.json", encoding="utf-8"))
    rel = json.load(open(src / "rel.json", encoding="utf-8"))
    imgs = tr["images"][: a.n_train]
    ids = {i["id"] for i in imgs}
    sub = {"images": imgs, "annotations": [x for x in tr["annotations"] if x["image_id"] in ids], "categories": tr["categories"]}
    ann = W / "reltr"
    ann.mkdir()
    for fn, d in (("train.json", sub), ("val.json", val), ("test.json", sub)):
        json.dump(d, open(ann / fn, "w", encoding="utf-8"))
    rs = {str(i): rel["train"][str(i)] for i in ids}
    json.dump({"rel_categories": rel["rel_categories"], "train": rs, "val": rel["val"], "test": rs}, open(ann / "rel.json", "w"))
    dev = [l.strip() for l in open(src / "dev300.txt", encoding="utf-8") if l.strip()][: a.n_dev]
    (ann / "dev300.txt").write_text("\n".join(dev) + "\n")
    shutil.copy(src / "meta.json", ann / "meta.json")
    frames = W / "frames"
    rng = np.random.default_rng(0)
    by_name = {i["file_name"]: i for i in val["images"] + imgs}
    for fn in [i["file_name"] for i in imgs] + [f[:-4] + ".jpg" for f in dev]:
        im = by_name[fn]
        p = frames / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (im["height"], im["width"], 3), dtype=np.uint8)).save(p, quality=85)

    py = sys.executable
    ckpt = ROOT / "data" / "ckpt" / "checkpoint0149.pth"
    base = [py, "main.py", "--dataset", "ag", "--ann_path", ann, "--img_folder", frames, "--device", "cpu",
            "--batch_size", "2", "--epochs", "2", "--lr_drop", "1", "--lr_backbone", "0", "--num_workers", "0",
            "--ag_ann_dir", ROOT / "data" / "ag" / "annotations", "--ckpt_every_min", "0", "--print_freq", "1", "--seed", "42",
            "--g7_no_stop"]  # ảnh nhiễu cho recall dev = 0: P13 sẽ dừng đúng như thiết kế, nên tắt riêng trong smoke test
    R = W / "runs"
    t0 = time.time()
    # tham chiếu: liền 6 vòng lặp (qua ranh giới epoch ở step 4 -> P13 + epoch1_weights)
    run(base + ["--output_dir", R / "ref", "--pretrained", ckpt, "--max_iters", "6", "--log_sample_ids"], env)
    # ngắt ở step 3 rồi resume tới 6
    run(base + ["--output_dir", R / "res", "--pretrained", ckpt, "--max_iters", "3", "--log_sample_ids"], env)
    run(base + ["--output_dir", R / "res", "--resume", R / "res" / "ckpt" / "last.pth", "--max_iters", "6", "--log_sample_ids"], env)
    # dừng đúng ở vòng lặp cuối của epoch 0 (step 4 = 4 vòng lặp/epoch) rồi resume: epoch còn 0 vòng lặp
    run(base + ["--output_dir", R / "edge", "--pretrained", ckpt, "--max_iters", "4"], env)
    run(base + ["--output_dir", R / "edge", "--resume", R / "edge" / "ckpt" / "last.pth", "--max_iters", "6"], env)
    st_edge = json.loads((R / "edge" / "STATUS.json").read_text(encoding="utf-8"))
    edge_ok = (st_edge["global_step"] == 6 and st_edge.get("epochs_completed") == 1
               and "epoch1_weights.pth" in st_edge.get("weights_sha256", {}))
    # hết giờ ngay từ đầu -> lưu, time_limit, mã 0
    rc_dl, _ = run(base + ["--output_dir", R / "dl", "--pretrained", ckpt, "--deadline_unix", str(time.time() - 1)], env, check=False)
    # hỏng last.pth -> restore_state chọn last_prev.pth
    cdir = W / "corrupt" / "runs" / "A1"
    shutil.copytree(R / "res", cdir)
    raw = (cdir / "ckpt" / "last.pth").read_bytes()
    (cdir / "ckpt" / "last.pth").write_bytes(raw[: len(raw) // 2])
    _, out = run([py, TOOLS / "restore_state.py", "--search", W / "corrupt", "--dst", W / "corrupt_dst", "--tmp", W / "unz"], env)
    (W / "corrupt_restore.txt").write_text(out)
    rc_g6, out = run([py, TOOLS / "g6_check.py", "--ref", R / "ref", "--res", R / "res", "--resume_step", "3", "--final_step", "6",
                      "--deadline_run", R / "dl", "--deadline_rc", str(rc_dl), "--corrupt_json", W / "corrupt_restore.txt",
                      "--readonly_rc", "0", "--out", W / "g6.json"], env, check=False)
    print(out[-1200:])
    # chạy tiếp res tới hết 2 epoch -> done
    run(base + ["--output_dir", R / "res", "--resume", R / "res" / "ckpt" / "last.pth"], env)
    run([py, TOOLS / "finalize_session.py", "--run", R / "res", "--t0", str(t0), "--rc", "0", "--wait_push_min", "0"], env)
    _, out = run([py, TOOLS / "restore_state.py", "--search", R, "--dst", W / "restored", "--tmp", W / "unz2"], env)
    last = json.loads(out.strip().splitlines()[-1])
    st = json.loads((R / "res" / "STATUS.json").read_text(encoding="utf-8"))
    g7 = json.loads((R / "res" / "checks" / "g7_dev_e1.json").read_text(encoding="utf-8"))
    checks = {
        "restore_done": last["mode"] == "done",
        "status_done": st["state"] == "done" and st["global_step"] == 8,
        "weights_sha": set(st["weights_sha256"]) == {"epoch1_weights.pth", "epoch2_weights.pth"},
        "g7_dev_file": g7["n_frames"]["full"] == a.n_dev,
        "sessions_csv": (R / "res" / "sessions.csv").exists(),
        "g6_check (ngoài phần đọc-chỉ và đẩy)": rc_g6 == 0,
        "resume đúng tại cuối epoch": edge_ok,
    }
    for k, v in checks.items():
        print(("ok  " if v else "SAI ") + k)
    print(f"smoke test {'QUA' if all(checks.values()) else 'KHÔNG QUA'} sau {(time.time() - t0) / 60:.1f} phút")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
