"""Phần dùng chung cho các script Action Genome (require mục 6).

Logic lọc frame và dựng nhãn thật chép từ STTran `dataloader/action_genome.py`
(github.com/yrcong/STTran, commit 6b559e1). Chạy được trên Linux (Kaggle) và Windows (máy nhà).
"""
import hashlib
import json
import pickle
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "1.0"
SEED = 42
DEV_RATIO = 0.05
DEV_EVAL_FRAMES = 300

# require 6.3 — chỉ số trong RelTR = chỉ số STTran; 0 là __background__
OBJ_CLASSES = [
    "__background__", "person", "bag", "bed", "blanket", "book", "box", "broom", "chair",
    "closet/cabinet", "clothes", "cup/glass/bottle", "dish", "door", "doorknob", "doorway",
    "floor", "food", "groceries", "laptop", "light", "medicine", "mirror", "paper/notebook",
    "phone/camera", "picture", "pillow", "refrigerator", "sandwich", "shelf", "shoe",
    "sofa/couch", "table", "television", "towel", "vacuum", "window",
]
# chỉ số STTran (0..25); id quan hệ trong RelTR = chỉ số STTran + 1 (0 là lớp giả)
REL_CLASSES = [
    "looking_at", "not_looking_at", "unsure",
    "above", "beneath", "in_front_of", "behind", "on_the_side_of", "in",
    "carrying", "covered_by", "drinking_from", "eating", "have_it_on_the_back", "holding",
    "leaning_on", "lying_on", "not_contacting", "other_relationship", "sitting_on",
    "standing_on", "touching", "twisting", "wearing", "wiping", "writing_on",
]
ATT = list(range(0, 3))
SPA = list(range(3, 9))
CON = list(range(9, 26))
GROUPS = {"attention": ATT, "spatial": SPA, "contacting": CON}

_STTRAN_OBJ_RENAME = {9: "closet/cabinet", 11: "cup/glass/bottle", 23: "paper/notebook",
                      24: "phone/camera", 31: "sofa/couch"}
_STTRAN_REL_RENAME = {0: "looking_at", 1: "not_looking_at", 5: "in_front_of", 7: "on_the_side_of",
                      10: "covered_by", 11: "drinking_from", 13: "have_it_on_the_back", 15: "leaning_on",
                      16: "lying_on", 17: "not_contacting", 18: "other_relationship", 19: "sitting_on",
                      20: "standing_on", 25: "writing_on"}

PKL = {"full": "object_bbox_and_relationship.pkl",
       "filtersmall": "object_bbox_and_relationship_filtersmall.pkl"}

# require 6.2 — số tham chiếu, đo ngày 2026-10-11
REF_COUNTS = {
    ("full", "train"): (7584, 177135, 376483),
    ("full", "test"): (1750, 56880, 146559),
    ("filtersmall", "train"): (7464, 166785, 324974),
    ("filtersmall", "test"): (1737, 54371, 126037),
}


def load_class_lists(ann_dir):
    """Đọc 2 file txt, đổi tên như STTran, assert khớp bảng require 6.3."""
    ann_dir = Path(ann_dir)
    obj = ["__background__"] + [l.strip("\n") for l in open(ann_dir / "object_classes.txt", encoding="utf-8")]
    for i, n in _STTRAN_OBJ_RENAME.items():
        obj[i] = n
    rel = [l.strip("\n") for l in open(ann_dir / "relationship_classes.txt", encoding="utf-8")]
    for i, n in _STTRAN_REL_RENAME.items():
        rel[i] = n
    assert obj == OBJ_CLASSES, f"object_classes.txt khác require 6.3: {obj}"
    assert rel == REL_CLASSES, f"relationship_classes.txt khác require 6.3: {rel}"
    return obj, rel


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ag(ann_dir, variant="full"):
    """Trả về (person_bbox, object_bbox) như STTran (filtersmall chỉ thay file vật)."""
    ann_dir = Path(ann_dir)
    load_class_lists(ann_dir)
    person = load_pkl(ann_dir / "person_bbox.pkl")
    obj = load_pkl(ann_dir / PKL[variant])
    return person, obj


def sttran_filter(person, obj, mode):
    """require 6.2 bước 1–4: OrderedDict video -> list frame (giữ thứ tự duyệt của STTran)."""
    video_dict = OrderedDict()
    for f in person.keys():
        if obj[f][0]["metadata"]["set"] == mode and any(o["visible"] for o in obj[f]):
            video_dict.setdefault(f.split("/")[0], []).append(f)
    out = OrderedDict()
    for v, frames in video_dict.items():
        keep = [f for f in frames if person[f]["bbox"].shape[0] > 0]
        if len(keep) > 2:
            out[v] = keep
    return out


def count_split(videos, obj):
    n_frames = sum(len(fs) for fs in videos.values())
    n_obj = sum(1 for fs in videos.values() for f in fs for o in obj[f] if o["visible"])
    return len(videos), n_frames, n_obj


def check_ref_counts(variant, mode, videos, obj):
    got = count_split(videos, obj)
    exp = REF_COUNTS.get((variant, mode))
    assert exp is None or got == exp, f"{variant}/{mode}: đếm được {got}, require 6.2 ghi {exp}"
    return got


def split_dev(train_videos, ratio=DEV_RATIO, seed=SEED):
    """require 6.6: 5% video train, chọn bằng seed 42 rồi sắp theo tên."""
    names = sorted(train_videos)
    n_dev = int(round(ratio * len(names)))
    dev = sorted(random.Random(seed).sample(names, n_dev))
    return dev


def even_frames(frames, k):
    """k frame cách đều trong danh sách frame hợp lệ (đã sắp theo chỉ số)."""
    frames = sorted(frames, key=frame_index)
    if len(frames) <= k:
        return frames
    idx = np.unique(np.linspace(0, len(frames) - 1, k).round().astype(int))
    return [frames[i] for i in idx]


def dev_eval_frames(dev_frames, n=DEV_EVAL_FRAMES, seed=SEED):
    """300 frame dev cố định cho P13 và G8."""
    pool = sorted(dev_frames)
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def frame_index(frame_key):
    return int(frame_key.split("/")[1].split(".")[0])


def jpg_name(frame_key):
    """'<VIDEO>.mp4/000089.png' -> '<VIDEO>.mp4/000089.jpg'."""
    return frame_key[:-4] + ".jpg"


def png_key(file_name):
    """Ngược lại của jpg_name; chấp nhận cả khi đã là .png."""
    return file_name[:-4] + ".png"


def build_gt_frame(person, obj, f):
    """Nhãn thật một frame theo logic STTran evaluation_recall.py.

    Trả về dict: boxes (N,4) xyxy theo pixel ảnh gốc (box 0 là người), classes (N,),
    relations (R,3) = [sub, obj, rel] với rel là chỉ số STTran 0..25; spatial ghi <vật, người>.
    """
    boxes = [np.asarray(person[f]["bbox"][0], dtype=float)]
    classes = [1]
    rels = []
    for o in obj[f]:
        if not o["visible"]:
            continue
        assert o["bbox"] is not None, f"{f}: vật visible mà không có bbox"
        x, y, w, h = o["bbox"]
        j = len(boxes)
        boxes.append(np.array([x, y, x + w, y + h], dtype=float))
        classes.append(OBJ_CLASSES.index(o["class"]))
        for r in o["attention_relationship"]:
            rels.append([0, j, REL_CLASSES.index(r)])
        for r in o["spatial_relationship"]:
            rels.append([j, 0, REL_CLASSES.index(r)])
        for r in o["contacting_relationship"]:
            rels.append([0, j, REL_CLASSES.index(r)])
    return {"boxes": np.stack(boxes), "classes": np.array(classes), "relations": np.array(rels, dtype=int).reshape(-1, 3)}


def gt_for_split(ann_dir, variant, mode, frame_filter=None):
    """dict frame_key -> nhãn thật, cho split 'train' | 'test' | 'dev' (dev lấy từ train)."""
    person, obj = load_ag(ann_dir, variant)
    base_mode = "train" if mode == "dev" else mode
    videos = sttran_filter(person, obj, base_mode)
    if mode == "dev":
        # tập dev luôn tách trên AG-full để hai biến thể có cùng danh sách video
        full_videos = videos if variant == "full" else sttran_filter(person, load_pkl(Path(ann_dir) / PKL["full"]), "train")
        dev = set(split_dev(full_videos))
        videos = OrderedDict((v, fs) for v, fs in videos.items() if v in dev)
    frames = [f for fs in videos.values() for f in fs]
    if frame_filter is not None:
        frames = [f for f in frames if f in frame_filter]
    return {f: build_gt_frame(person, obj, f) for f in frames}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dump_json(obj, path, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    tmp.replace(path)


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def deadline_passed(deadline_unix, margin=0):
    return deadline_unix is not None and deadline_unix > 0 and time.time() + margin >= deadline_unix
