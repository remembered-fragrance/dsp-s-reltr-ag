"""FR-2: chuyển nhãn AG sang định dạng COCO + rel.json của RelTR (require 6.4, 6.6).

Đầu ra trong --out_dir: train.json, val.json (= dev), test.json (AG-full), rel.json, meta.json, dev300.txt.
Với --overfit 64: thêm <out_dir>_overfit/ có train = val = test = 64 frame train cố định.

Ví dụ (máy nhà, chỉ có frame test + dev):
  python tools/convert_ag.py --ann_dir data/ag/annotations --frames_dir data/frames --out_dir data/reltr --splits test dev
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_common as C  # noqa: E402

K_MAX = 8


def clip_xyxy(b, w, h):
    x0, y0, x1, y1 = [float(v) for v in b]
    x0, x1 = min(max(x0, 0.0), w), min(max(x1, 0.0), w)
    y0, y1 = min(max(y0, 0.0), h), min(max(y1, 0.0), h)
    return x0, y0, x1, y1


def frame_record(f, person, obj, stats):
    """Danh sách annotation (person đầu tiên) và bộ ba <person, rel, vật>; None nếu bỏ frame."""
    w, h = person[f]["bbox_size"]
    px0, py0, px1, py1 = clip_xyxy(person[f]["bbox"][0], w, h)
    if px1 - px0 < 1 or py1 - py0 < 1:
        stats["drop_frame_degenerate_person"] += 1
        return None
    anns = [(1, [px0, py0, px1 - px0, py1 - py0])]
    rels = []
    for o in obj[f]:
        if not o["visible"]:
            continue
        x, y, bw, bh = o["bbox"]
        x0, y0, x1, y1 = clip_xyxy([x, y, x + bw, y + bh], w, h)
        if x1 - x0 < 1 or y1 - y0 < 1:
            stats["drop_box_degenerate"] += 1
            continue
        cls = C.OBJ_CLASSES.index(o["class"])  # ValueError nếu nhãn lạ -> dừng
        j = len(anns)
        anns.append((cls, [x0, y0, x1 - x0, y1 - y0]))
        att = o["attention_relationship"]
        if len(att) != 1:
            stats["attention_not_single"] += 1
        for group, key in (("attention", "attention_relationship"), ("spatial", "spatial_relationship"),
                           ("contacting", "contacting_relationship")):
            for r in o[key]:
                rid = C.REL_CLASSES.index(r)
                assert rid in C.GROUPS[group], f"{f}: {r} không thuộc nhóm {group}"
                rels.append([0, j, rid + 1])  # id RelTR = chỉ số STTran + 1; mọi bộ ba là <person, rel, vật>
                stats[f"triplets_{group}"] += 1
    if len(anns) == 1:
        stats["drop_frame_no_object_after_filter"] += 1
        return None
    return (w, h), anns, rels


def build_split(name, frames, person, obj, frames_dir, check_size, stats, check_frames=True):
    images, annotations, rel = [], [], {}
    missing, size_bad = [], []
    aid = 1
    for f in sorted(frames):
        rec = frame_record(f, person, obj, stats)
        if rec is None:
            continue
        (w, h), anns, rels = rec
        fn = C.jpg_name(f)
        p = Path(frames_dir) / fn
        if check_frames and not p.exists():
            missing.append(fn)
            continue
        if check_frames and check_size:
            with Image.open(p) as im:
                if im.size != (w, h):
                    size_bad.append((fn, im.size, (w, h)))
        iid = len(images) + 1
        images.append({"id": iid, "file_name": fn, "width": int(w), "height": int(h)})
        for cls, (x, y, bw, bh) in anns:
            annotations.append({"id": aid, "image_id": iid, "category_id": cls, "bbox": [x, y, bw, bh],
                                "area": bw * bh, "iscrowd": 0})
            aid += 1
        rel[str(iid)] = rels
    assert not missing, f"{name}: thiếu {len(missing)} frame, ví dụ {missing[:5]}"
    assert not size_bad, f"{name}: {len(size_bad)} ảnh lệch kích thước, ví dụ {size_bad[:3]}"
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": i, "name": n} for i, n in enumerate(C.OBJ_CLASSES) if i > 0]}
    return coco, rel


def validate(coco, rel):
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
        assert a["bbox"][2] >= 1 and a["bbox"][3] >= 1, f"box suy biến: {a}"
        assert 1 <= a["category_id"] <= 36
    for im in coco["images"]:
        anns = by_img[im["id"]]
        assert anns[0]["category_id"] == 1, f"annotation đầu của ảnh {im['id']} không phải person"
        assert sum(a["category_id"] == 1 for a in anns) == 1
        for s, o, p in rel[str(im["id"])]:
            assert s == 0 and 0 < o < len(anns), f"chỉ số quan hệ sai ở ảnh {im['id']}: {(s, o, p)}"
            assert 1 <= p <= 26
        assert rel[str(im["id"])], f"ảnh {im['id']} không có bộ ba nào"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=K_MAX)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"], choices=["train", "dev", "test"])
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--no_size_check", action="store_true")
    ap.add_argument("--skip_frames_check", action="store_true", help="CHỈ để thử nghiệm khi chưa có frame; ghi vào meta.json")
    args = ap.parse_args()
    assert 1 <= args.k <= K_MAX
    out = Path(args.out_dir)
    for bad in ("train", "val"):
        assert bad not in out.as_posix().lower(), f"--out_dir không được chứa '{bad}' (datasets/coco.py chọn split theo chuỗi)"

    person, obj = C.load_ag(args.ann_dir, "full")
    train_v = C.sttran_filter(person, obj, "train")
    test_v = C.sttran_filter(person, obj, "test")
    counts = {"full": {"train": C.check_ref_counts("full", "train", train_v, obj),
                       "test": C.check_ref_counts("full", "test", test_v, obj)}}
    assert not set(train_v) & set(test_v), "có video vừa ở train vừa ở test"

    # biến thể phụ (MR-12): chỉ để đếm và assert tập con
    obj_fs = C.load_pkl(Path(args.ann_dir) / C.PKL["filtersmall"])
    test_fs = C.sttran_filter(person, obj_fs, "test")
    counts["filtersmall"] = {"train": C.check_ref_counts("filtersmall", "train", C.sttran_filter(person, obj_fs, "train"), obj_fs),
                             "test": C.check_ref_counts("filtersmall", "test", test_fs, obj_fs)}
    full_test_frames = {f for fs in test_v.values() for f in fs}
    assert {f for fs in test_fs.values() for f in fs} <= full_test_frames, "AG-filtersmall không là tập con của AG-full"
    del obj_fs

    dev_videos = C.split_dev(train_v)
    dev_set = set(dev_videos)
    split_frames = {
        "train": [f for v, fs in train_v.items() if v not in dev_set for f in C.even_frames(C.even_frames(fs, K_MAX), args.k)],
        "dev": [f for v in dev_videos for f in train_v[v]],
        "test": sorted(full_test_frames),
    }
    file_of = {"train": "train.json", "dev": "val.json", "test": "test.json"}
    out.mkdir(parents=True, exist_ok=True)
    stats = {s: Counter() for s in args.splits}
    rel_all = {"rel_categories": ["__background__"] + C.REL_CLASSES}
    meta_splits, cocos = {}, {}
    for s in args.splits:
        coco, rel = build_split(s, split_frames[s], person, obj, args.frames_dir, not args.no_size_check, stats[s],
                                check_frames=not args.skip_frames_check)
        validate(coco, rel)
        cocos[s] = coco
        rel_all[{"train": "train", "dev": "val", "test": "test"}[s]] = rel
        C.dump_json(coco, out / file_of[s])
        vids = {im["file_name"].split("/")[0] for im in coco["images"]}
        meta_splits[s] = {"videos": len(vids), "frames": len(coco["images"]),
                          "boxes": len(coco["annotations"]), "triplets": sum(len(r) for r in rel.values()),
                          "stats": dict(stats[s]), "sha256": C.sha256_file(out / file_of[s])}
        print(f"{s}: {meta_splits[s]}", flush=True)
    if "test" in args.splits:
        assert meta_splits["test"]["frames"] == C.REF_COUNTS[("full", "test")][1], "số frame test khác bộ lọc STTran"
    C.dump_json(rel_all, out / "rel.json")

    dev300 = C.dev_eval_frames(split_frames["dev"])
    (out / "dev300.txt").write_text("\n".join(dev300) + "\n", encoding="utf-8")
    meta = {"script": "convert_ag.py", "version": C.SCRIPT_VERSION, "created": C.now_str(), "k": args.k, "k_max": K_MAX,
            "seed": C.SEED, "dev_ratio": C.DEV_RATIO, "splits": meta_splits, "sttran_counts": counts,
            "dev_videos": dev_videos, "dev300_sha256": C.sha256_file(out / "dev300.txt"),
            "rel_json_sha256": C.sha256_file(out / "rel.json"), "frames_checked": not args.skip_frames_check,
            "note": "train.json chỉ sinh khi --splits có train; nhãn thật để chấm dựng lại từ pkl (MR-7)"}
    C.dump_json(meta, out / "meta.json", indent=1)

    if args.overfit:
        assert "train" in args.splits, "--overfit cần split train"
        imgs = cocos["train"]["images"]
        pick = sorted(random.Random(C.SEED).sample(range(len(imgs)), args.overfit))
        ids = {imgs[i]["id"] for i in pick}
        sub = {"images": [im for im in imgs if im["id"] in ids],
               "annotations": [a for a in cocos["train"]["annotations"] if a["image_id"] in ids],
               "categories": cocos["train"]["categories"]}
        rel_sub = {str(i): rel_all["train"][str(i)] for i in sorted(ids)}
        od = out.parent / (out.name + "_overfit")
        od.mkdir(parents=True, exist_ok=True)
        for fn in ("train.json", "val.json", "test.json"):
            C.dump_json(sub, od / fn)
        C.dump_json({"rel_categories": rel_all["rel_categories"], "train": rel_sub, "val": rel_sub, "test": rel_sub},
                    od / "rel.json")
        # danh sách frame (khóa .png của AG) để G4 chấm lại bằng ag_eval --split train --frame_list
        (od / "overfit_frames.txt").write_text("\n".join(C.png_key(im["file_name"]) for im in sub["images"]) + "\n",
                                               encoding="utf-8")
        print(f"overfit: {len(sub['images'])} frame -> {od}")
    print("convert_ag: xong, mọi assert đều qua")


if __name__ == "__main__":
    main()
