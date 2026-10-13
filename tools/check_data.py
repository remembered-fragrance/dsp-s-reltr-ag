"""FR-3 / G2: vẽ 20 frame ngẫu nhiên (seed cố định) kèm box, nhãn, bộ ba; in thống kê.

  python tools/check_data.py --ann_path data/reltr --img_folder data/frames --split test --out_dir checks/data_viz --stats checks/g2_stats.txt
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_common as C  # noqa: E402

SPLIT_FILE = {"train": "train.json", "dev": "val.json", "test": "test.json"}
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_path", required=True)
    ap.add_argument("--img_folder", required=True)
    ap.add_argument("--split", default="test", choices=list(SPLIT_FILE))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stats", required=True)
    args = ap.parse_args()
    ann = Path(args.ann_path)
    coco = json.load(open(ann / SPLIT_FILE[args.split], encoding="utf-8"))
    rel = json.load(open(ann / "rel.json", encoding="utf-8"))[{"train": "train", "dev": "val", "test": "test"}[args.split]]
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    rel_names = rel_cats = json.load(open(ann / "rel.json", encoding="utf-8"))["rel_categories"]

    n_img = len(coco["images"])
    n_obj = sum(len(v) - 1 for v in by_img.values())
    grp = Counter()
    per_rel = Counter()
    for rs in rel.values():
        for _, _, p in rs:
            rid = p - 1
            grp[next(g for g, ix in C.GROUPS.items() if rid in ix)] += 1
            per_rel[rel_cats[p]] += 1
    lines = [f"split {args.split}: {n_img} frame, {len({im['file_name'].split('/')[0] for im in coco['images']})} video",
             f"vật trung bình mỗi frame: {n_obj / max(n_img, 1):.2f}",
             f"bộ ba theo nhóm: {dict(grp)}",
             f"bộ ba theo lớp: {dict(per_rel.most_common())}"]
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for im in random.Random(C.SEED).sample(coco["images"], min(args.n, n_img)):
        img = Image.open(Path(args.img_folder) / im["file_name"]).convert("RGB")
        scale = 2 if img.width < 600 else 1
        img = img.resize((img.width * scale, img.height * scale))
        dr = ImageDraw.Draw(img)
        anns = by_img[im["id"]]
        for i, a in enumerate(anns):
            x, y, w, h = [v * scale for v in a["bbox"]]
            c = COLORS[i % len(COLORS)]
            dr.rectangle([x, y, x + w, y + h], outline=c, width=2)
            dr.text((x + 3, y + 3), f"{i}:{C.OBJ_CLASSES[a['category_id']]}", fill=c)
        ty = 5
        for s, o, p in rel[str(im["id"])]:
            dr.text((5, ty), f"{s}:person -{rel_names[p]}-> {o}:{C.OBJ_CLASSES[anns[o]['category_id']]}", fill=(255, 255, 0))
            ty += 12
        img.save(out / (im["file_name"].replace("/", "__")))
    print(f"Đã vẽ {min(args.n, n_img)} frame vào {out}. Mở vài ảnh xem box có đúng vị trí không (G2).")


if __name__ == "__main__":
    main()
