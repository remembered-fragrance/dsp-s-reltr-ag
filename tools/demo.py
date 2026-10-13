"""FR-10: demo video -> đồ thị cảnh theo từng frame (máy nhà). Chỉ minh họa, không dùng số liệu demo trong báo cáo.

  python tools/demo.py --video <mp4> --ckpt runs/A1/ckpt/epoch4_weights.pth --fps 2 --out demo_out/<tên>
Đầu ra: annotated.mp4, graphs.json, summary.md trong --out.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import ag_common as C  # noqa: E402
import ag_eval as E  # noqa: E402
import ag_pred as P  # noqa: E402
from extract_frames import find_ffmpeg  # noqa: E402
from predict_ag import DEFAULT_RELTR, load_model  # noqa: E402

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (210, 245, 60), (250, 190, 212), (0, 128, 128)]
_MODEL = {}


def get_model(ckpt, device, reltr_dir=DEFAULT_RELTR):
    key = (str(ckpt), str(device))
    if key not in _MODEL:
        _MODEL.clear()
        _MODEL[key] = load_model(ckpt, device, reltr_dir)
    return _MODEL[key]


def frame_triplets(trip, thr, topk):
    """Bộ ba theo With Constraint (mỗi query: 1 attention, 1 spatial, 1 contacting), lọc điểm và gộp trùng."""
    e = E.adapt(trip)
    pr, ps = E.select_pred_rels("with", e["pred_rel_inds"], e["rel_scores"], e["row_group"], e["obj_scores"])
    score = e["obj_scores"][pr[:, 0]] * e["obj_scores"][pr[:, 1]] * ps
    q = trip.shape[0]
    out, seen = [], set()
    for j in np.argsort(-score):
        if score[j] < thr or len(out) >= topk:
            break
        a, b, r = (int(v) for v in pr[j])
        person, obj = (a, b) if a < q else (b, a)  # spatial được chấm theo chiều <vật, người>; hiển thị luôn từ người
        ocls = C.OBJ_CLASSES[int(e["pred_classes"][obj])]
        key = (C.REL_CLASSES[r], ocls, tuple(np.round(e["pred_boxes"][obj] / 20).astype(int)))
        if key in seen:
            continue
        seen.add(key)
        out.append({"subject": "person", "predicate": C.REL_CLASSES[r], "object": ocls, "score": round(float(score[j]), 3),
                    "group": ["attention", "spatial", "contacting"][int(e["row_group"][j])],
                    "sub_box": [round(float(v), 1) for v in e["pred_boxes"][person]],
                    "obj_box": [round(float(v), 1) for v in e["pred_boxes"][obj]]})
    return out


def draw(img, triplets):
    s = 2 if img.width < 640 else 1
    img = img.resize((img.width * s, img.height * s))
    dr = ImageDraw.Draw(img)
    colors = {}
    for t in triplets:
        c = colors.setdefault(t["object"], PALETTE[len(colors) % len(PALETTE)])
        dr.rectangle([v * s for v in t["sub_box"]], outline=(255, 255, 255), width=2)
        dr.rectangle([v * s for v in t["obj_box"]], outline=c, width=2)
        dr.text((t["obj_box"][0] * s + 3, t["obj_box"][1] * s + 3), t["object"], fill=c)
    y = 4
    for t in triplets:
        dr.rectangle([2, y - 1, 8 + 7 * len(t["predicate"] + t["object"]) + 90, y + 11], fill=(0, 0, 0))
        dr.text((4, y), f"person {t['predicate']} {t['object']} ({t['score']:.2f})", fill=colors[t["object"]])
        y += 13
    return img


@torch.no_grad()
def run(video, ckpt, out_dir, fps=2.0, thr=0.3, topk=10, device=None, amp=True, reltr_dir=DEFAULT_RELTR, progress=None,
        source=None, max_seconds=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, transform = get_model(ckpt, device, reltr_dir)  # thêm reltr_dir vào sys.path, nên import util của RelTR sau dòng này
    from util.misc import nested_tensor_from_tensor_list
    ffmpeg = find_ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="demo_"))
    try:
        (tmp / "in").mkdir()
        (tmp / "out").mkdir()
        cut = ["-t", str(max_seconds)] if max_seconds else []  # tùy chọn đầu vào: chỉ đọc max_seconds giây đầu
        subprocess.run([ffmpeg, "-loglevel", "error", *cut, "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2",
                        str(tmp / "in" / "%06d.jpg")], check=True)
        frames = sorted((tmp / "in").glob("*.jpg"))
        assert frames, "không đọc được frame nào từ video"
        graphs = []
        for i, fp in enumerate(frames):
            img = Image.open(fp).convert("RGB")
            x, _ = transform(img, None)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                o = model(nested_tensor_from_tensor_list([x.to(device)]))
            trip = P.pack_outputs(o, [img.size])[0]["trip"].float().numpy()
            tri = frame_triplets(trip, thr, topk)
            graphs.append({"frame": i + 1, "time_s": round(i / fps, 2), "triplets": tri})
            draw(img, tri).save(tmp / "out" / f"{i + 1:06d}.jpg", quality=92)
            if progress:
                progress((i + 1) / len(frames))
        mp4 = out_dir / "annotated.mp4"
        enc = [ffmpeg, "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "out" / "%06d.jpg"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"]
        if subprocess.run(enc + ["-c:v", "libx264", str(mp4)]).returncode != 0:
            subprocess.run(enc + ["-c:v", "mpeg4", "-q:v", "3", str(mp4)], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    meta = {"video": str(video), "source": source, "ckpt": str(ckpt), "fps": fps, "thr": thr, "topk": topk,
            "max_seconds": max_seconds, "device": str(device),
            "frames": len(graphs), "created": C.now_str(), "note": "Demo minh họa; không dùng số liệu demo trong báo cáo."}
    C.dump_json({"meta": meta, "frames": graphs}, out_dir / "graphs.json", indent=1)
    write_summary(out_dir / "summary.md", meta, graphs)
    return mp4, out_dir / "graphs.json", graphs


def write_summary(path, meta, graphs):
    seen = defaultdict(list)
    for g in graphs:
        for key in dict.fromkeys((t["predicate"], t["object"]) for t in g["triplets"]):  # một frame đếm một lần
            seen[key].append(g["time_s"])
    L = [f"# Demo: {Path(meta['video']).name}", "",
         f"Nguồn video: {meta.get('source') or 'không ghi'} (`{meta['video']}`); checkpoint `{meta['ckpt']}`; {meta['frames']} frame ở {meta['fps']} fps; "
         f"ngưỡng điểm {meta['thr']}, tối đa {meta['topk']} bộ ba mỗi frame.", "",
         "> Demo chỉ để minh họa; không dùng số liệu ở đây trong báo cáo kết quả.", "",
         "| Quan hệ | Lần đầu (s) | Lần cuối (s) | Số frame |", "|---|---|---|---|"]
    for (p, o), ts in sorted(seen.items(), key=lambda kv: (kv[1][0], -len(kv[1]))):
        L.append(f"| person — {p} — {o} | {ts[0]} | {ts[-1]} | {len(ts)} |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--reltr_dir", default=str(DEFAULT_RELTR))
    ap.add_argument("--source", help="nguồn video ghi vào summary.md (FR-10), vd 'Charades v1 480p, split test của AG'")
    a = ap.parse_args()
    mp4, gj, graphs = run(a.video, a.ckpt, a.out, a.fps, a.thr, a.topk, a.device, not a.no_amp, a.reltr_dir, source=a.source)
    print(f"{len(graphs)} frame -> {mp4}, {gj}")


if __name__ == "__main__":
    main()
