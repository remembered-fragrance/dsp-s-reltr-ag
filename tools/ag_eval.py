"""FR-6: đánh giá SGDet trên Action Genome theo giao thức STTran (chạy bằng CPU).

Port từ STTran lib/evaluation_recall.py (commit 6b559e1): evaluate_from_dict, evaluate_recall, _triplet,
_compute_pred_matches, bbox_overlaps (Cython, quy ước +1 pixel) — giữ nguyên logic. Khác biệt duy nhất:
  * Không softmax attention bên trong (như OED models/evaluate_recall.py): dự đoán đã là xác suất.
  * Semi Constraint xác định nhóm của dòng bằng nhãn nhóm thay vì kiểm tổng xác suất > 0 (cùng ý định,
    tránh sai khi xác suất float16 bị làm tròn về 0).
mR@K theo TEMPURA (sayaknag/unbiasedSGG): cộng hit/count theo lớp trên toàn tập rồi lấy trung bình 26 lớp.

Adapter A1 (mỗi query là một cặp): box người = box subject, điểm = p(person); box vật = box object, điểm = max p
lớp vật 2..36; quan hệ đã chuẩn hóa theo nhóm (xem ag_pred.py). Dễ hơn With Constraint thật một chút vì
nhiều query có thể cùng chỉ một cặp người–vật.

Ví dụ:
  python tools/ag_eval.py --pred runs/A1/pred_test.pt --ann_dir data/ag/annotations --split test \
      --gt full filtersmall --modes with semi no --iou 0.5 0.3 --out runs/A1/eval
  python tools/ag_eval.py --gt_as_pred --ann_dir data/ag/annotations --split test --gt full filtersmall --out checks/g3
"""
import argparse
import json
import os
import sys
import time
from functools import reduce
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_common as C  # noqa: E402
import ag_pred as P  # noqa: E402

KS = (10, 20, 50, 100)
N_REL = 26
SEMI_THRESHOLD = 0.9
GROUP_OF = np.array([0] * 3 + [1] * 6 + [2] * 17)


# ----------------------------------------------------------------- port từ STTran (giữ nguyên logic)
def bbox_overlaps(boxes, query_boxes):
    """lib/fpn/box_intersections_cpu/bbox.pyx, quy ước +1 pixel."""
    boxes = np.asarray(boxes, dtype=float)
    q = np.asarray(query_boxes, dtype=float)
    area_q = (q[:, 2] - q[:, 0] + 1) * (q[:, 3] - q[:, 1] + 1)
    area_b = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    iw = np.minimum(boxes[:, None, 2], q[None, :, 2]) - np.maximum(boxes[:, None, 0], q[None, :, 0]) + 1
    ih = np.minimum(boxes[:, None, 3], q[None, :, 3]) - np.maximum(boxes[:, None, 1], q[None, :, 1]) + 1
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    ua = area_b[:, None] + area_q[None, :] - inter
    return np.where(inter > 0, inter / ua, 0.0)


def intersect_2d(x1, x2):
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")
    return (x1[..., None] == x2.T[None, ...]).all(1)


def argsort_desc(scores):
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))


def _triplet(predicates, relations, classes, boxes, predicate_scores=None, class_scores=None):
    assert predicates.shape[0] == relations.shape[0]
    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))
    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((class_scores[relations[:, 0]], class_scores[relations[:, 1]], predicate_scores))
    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thresh):
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for _ in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0], gt_boxes[gt_has_match], keeps[gt_has_match]):
        boxes = pred_boxes[keep_inds]
        sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
        obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]
        inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)
        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def evaluate_recall(gt_rels, gt_boxes, gt_classes, pred_rels, pred_boxes, pred_classes,
                    rel_scores=None, cls_scores=None, iou_thresh=0.5):
    if pred_rels.size == 0:
        return [[]]
    assert gt_rels.shape[0] != 0
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2], gt_rels[:, :2], gt_classes, gt_boxes)
    assert pred_rels[:, :2].max() < pred_classes.shape[0]
    pred_triplets, pred_triplet_boxes, relation_scores = _triplet(
        pred_rels[:, 2], pred_rels[:, :2], pred_classes, pred_boxes, rel_scores, cls_scores)
    order = relation_scores.prod(1).argsort()[::-1]
    return _compute_pred_matches(gt_triplets, pred_triplets[order], gt_triplet_boxes, pred_triplet_boxes[order], iou_thresh)


def select_pred_rels(method, pred_rel_inds, rel_scores, row_group, obj_scores):
    if method == "semi":
        pred_rels, predicate_scores = [], []
        for i, j in enumerate(pred_rel_inds):
            if row_group[i] == 0:  # attention: luôn lấy 1
                pred_rels.append(np.append(j, rel_scores[i].argmax()))
                predicate_scores.append(rel_scores[i].max())
            else:                   # spatial / contacting: mọi lớp có điểm > 0,9
                for k in np.where(rel_scores[i] > SEMI_THRESHOLD)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i, k])
        return np.array(pred_rels).reshape(-1, 3).astype(int), np.array(predicate_scores, dtype=float)
    if method == "no":
        overall = obj_scores[pred_rel_inds].prod(1)[:, None] * rel_scores
        score_inds = argsort_desc(overall)[:100]
        return (np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1])),
                rel_scores[score_inds[:, 0], score_inds[:, 1]])
    return np.column_stack((pred_rel_inds, rel_scores.argmax(1))), rel_scores.max(1)  # with


# ----------------------------------------------------------------- adapter A1
def adapt(trip):
    """trip: (Q, 37) float -> pred_entry kiểu STTran (mỗi query một cặp, box riêng)."""
    trip = np.asarray(trip, dtype=float)
    q = trip.shape[0]
    boxes = np.concatenate([trip[:, P.C_SUB], trip[:, P.C_OBJ]], 0)
    classes = np.concatenate([np.full(q, P.PERSON), trip[:, P.C_OL].round().astype(int)])
    scores = np.concatenate([trip[:, P.C_PP], trip[:, P.C_OS]])
    pair = np.stack([np.arange(q), q + np.arange(q)], 1)
    rel = trip[:, P.C_REL]
    blocks, inds, groups = [], [], []
    for g, s in enumerate(P.GROUP_SLICES):
        m = np.zeros_like(rel)
        m[:, s] = rel[:, s]
        blocks.append(m)
        inds.append(pair[:, ::-1] if g == 1 else pair)  # spatial: <vật, người>
        groups.append(np.full(q, g))
    return {"pred_boxes": boxes, "pred_classes": classes, "obj_scores": scores,
            "pred_rel_inds": np.concatenate(inds), "rel_scores": np.concatenate(blocks), "row_group": np.concatenate(groups)}


def gt_to_trip(gt):
    """--gt_as_pred: mỗi vật thật thành một query; nhãn thật có điểm 1, còn lại 0."""
    rows = []
    rels = gt["relations"]
    for j in range(1, len(gt["classes"])):
        r = np.zeros(P.N_TRIP_COLS)
        r[P.C_SUB], r[P.C_OBJ] = gt["boxes"][0], gt["boxes"][j]
        r[P.C_PP], r[P.C_OL], r[P.C_OS] = 1.0, gt["classes"][j], 1.0
        mine = rels[((rels[:, 0] == 0) & (rels[:, 1] == j)) | ((rels[:, 0] == j) & (rels[:, 1] == 0))]
        r[P.C_REL.start + mine[:, 2]] = 1.0
        rows.append(r)
    return np.stack(rows).astype(np.float16)


# ----------------------------------------------------------------- chấm một frame
def eval_frame(args):
    gt, trip, modes, ious = args
    entry = adapt(trip)
    gt_rels, gt_boxes, gt_classes = gt["relations"], gt["boxes"].astype(float), gt["classes"]
    labels = gt_rels[:, 2]
    out = {}
    for m in modes:
        pr, ps = select_pred_rels(m, entry["pred_rel_inds"], entry["rel_scores"], entry["row_group"], entry["obj_scores"])
        for iou in ious:
            p2g = evaluate_recall(gt_rels, gt_boxes, gt_classes, pr, entry["pred_boxes"], entry["pred_classes"],
                                  ps, entry["obj_scores"], iou_thresh=iou)
            res = {}
            for k in KS:
                match = reduce(np.union1d, p2g[:k]) if len(p2g) else np.array([])
                hit = np.bincount(labels[np.asarray(match, dtype=int)], minlength=N_REL)
                res[k] = (len(match) / len(gt_rels), hit)
            out[(m, iou)] = res
    return out, np.bincount(labels, minlength=N_REL)


def summarize(per_frame, modes, ious):
    metrics = {}
    total_count = sum(c for _, c in per_frame)
    for m in modes:
        for iou in ious:
            d = {}
            for k in KS:
                rec = np.mean([r[(m, iou)][k][0] for r, _ in per_frame])
                hits = sum(r[(m, iou)][k][1] for r, _ in per_frame)
                cls_rec = hits / np.maximum(total_count, 1e-10)
                d[f"R@{k}"] = 100 * rec
                d[f"mR@{k}"] = 100 * float(np.mean(hits / (total_count + 1e-10)))
                d[f"per_class@{k}"] = {C.REL_CLASSES[i]: 100 * float(cls_rec[i]) for i in range(N_REL)}
                d[f"per_group@{k}"] = {g: 100 * float(hits[ix].sum() / max(total_count[ix].sum(), 1))
                                       for g, ix in C.GROUPS.items()}
            metrics[f"{m}@{iou}"] = d
    return metrics, total_count


def load_pred(path):
    import torch
    pred = torch.load(path, map_location="cpu", weights_only=False)
    return pred.get("frames", pred)


def write_md(path, results, counts, n_frames, args):
    lines = [f"# Kết quả SGDet — split {args.split}", "",
             f"Số frame: {n_frames}. Nguồn dự đoán: `{'gt_as_pred' if args.gt_as_pred else args.pred}`.", ""]
    for variant, metrics in results.items():
        lines += [f"## Nhãn thật {variant}", "", "| Chế độ | IoU | R@10 | R@20 | R@50 | mR@10 | mR@20 | mR@50 |", "|---|---|---|---|---|---|---|---|"]
        for key, d in metrics.items():
            m, iou = key.split("@")
            lines.append(f"| {m} | {iou} | " + " | ".join(f"{d[f'R@{k}']:.2f}" for k in (10, 20, 50)) + " | "
                         + " | ".join(f"{d[f'mR@{k}']:.2f}" for k in (10, 20, 50)) + " |")
        key0 = next((k for k in metrics if k.startswith("with@0.5")), next(iter(metrics)))
        d = metrics[key0]
        lines += ["", f"Recall theo nhóm ({key0}, R@20): " + ", ".join(f"{g} {v:.2f}" for g, v in d["per_group@20"].items()), ""]
        cnt = counts[variant]
        rare = sorted(C.CON, key=lambda i: cnt[i])[:5]
        lines += [f"5 lớp contacting hiếm nhất ({key0}, R@20): " + ", ".join(
            f"{C.REL_CLASSES[i]} (n={int(cnt[i])}) {d['per_class@20'][C.REL_CLASSES[i]]:.2f}" for i in rare), ""]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--gt_as_pred", action="store_true")
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--split", default="test", choices=["test", "dev", "train"])
    ap.add_argument("--gt", nargs="+", default=["full", "filtersmall"], choices=["full", "filtersmall"])
    ap.add_argument("--modes", nargs="+", default=["with", "semi", "no"], choices=["with", "semi", "no"])
    ap.add_argument("--iou", nargs="+", type=float, default=[0.5, 0.3])
    ap.add_argument("--frame_list", help="file txt, mỗi dòng một frame (.png hoặc .jpg)")
    ap.add_argument("--meta", help="meta.json của convert_ag: kiểm danh sách video dev")
    ap.add_argument("--compare", help="g7_dev_e*.json để so (G8)")
    ap.add_argument("--max_frames", type=int, default=0, help="chỉ để thử nghiệm")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.gt_as_pred or args.pred, "cần --pred hoặc --gt_as_pred"
    t0 = time.time()

    keep = None
    if args.frame_list:
        keep = {C.png_key(l.strip()) for l in open(args.frame_list, encoding="utf-8") if l.strip()}
    pred = None if args.gt_as_pred else load_pred(args.pred)
    if args.meta and args.split == "dev":
        meta = json.load(open(args.meta, encoding="utf-8"))
        person, obj = C.load_ag(args.ann_dir, "full")
        assert meta["dev_videos"] == C.split_dev(C.sttran_filter(person, obj, "train")), "danh sách video dev khác meta.json"

    results, counts, n_frames = {}, {}, {}
    for variant in args.gt:
        gts = C.gt_for_split(args.ann_dir, variant, args.split, keep)
        frames = sorted(gts)
        if args.max_frames:
            frames = frames[: args.max_frames]
        if keep is not None:
            assert len(gts) == len(keep), f"{len(keep) - len(gts)} frame trong --frame_list không có nhãn thật ({variant})"
        if pred is not None:
            miss = [f for f in frames if f not in pred]
            assert not miss, f"{len(miss)} frame có nhãn thật nhưng thiếu dự đoán, ví dụ {miss[:3]}"
        jobs = [(gts[f], gt_to_trip(gts[f]) if pred is None else pred[f]["trip"].float().numpy(), args.modes, args.iou)
                for f in frames]
        with Pool(args.workers) as pool:
            per_frame = pool.map(eval_frame, jobs, chunksize=64)
        results[variant], counts[variant] = summarize(per_frame, args.modes, args.iou)
        n_frames[variant] = len(frames)
        if ("with", 0.5) in per_frame[0][0]:  # recall từng frame, để make_report chọn ví dụ tốt/tệ
            per_frame_r = {f: {"R@20_with": r[("with", 0.5)][20][0], "n_gt": int(c.sum())} for f, (r, c) in zip(frames, per_frame)}
            Path(args.out).mkdir(parents=True, exist_ok=True)
            C.dump_json(per_frame_r, Path(args.out) / f"per_frame_{variant}.json")
        d = results[variant]
        print(f"[{variant}] {len(frames)} frame; " + "; ".join(
            f"{k} R@20={v['R@20']:.2f} mR@20={v['mR@20']:.2f}" for k, v in d.items()), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"split": args.split, "gt_as_pred": args.gt_as_pred, "pred": args.pred, "frame_list": args.frame_list,
               "n_frames": n_frames, "modes": args.modes, "iou": args.iou, "semi_threshold": SEMI_THRESHOLD,
               "gt_triplet_counts": {v: {C.REL_CLASSES[i]: int(c[i]) for i in range(N_REL)} for v, c in counts.items()},
               "metrics": results, "seconds": round(time.time() - t0, 1), "time": C.now_str()}
    if args.compare:
        ref = json.load(open(args.compare, encoding="utf-8"))
        diffs = {}
        for key in ("with@0.5", "no@0.5"):
            if key in results.get("full", {}) and key in ref.get("metrics", {}).get("full", {}):
                diffs[key] = results["full"][key]["R@20"] - ref["metrics"]["full"][key]["R@20"]
        payload["compare"] = {"ref": args.compare, "delta_R@20": diffs, "pass": bool(diffs) and all(abs(v) <= 1.0 for v in diffs.values())}
        print("So sánh:", payload["compare"])
    C.dump_json(payload, out / "metrics.json", indent=1)
    write_md(out / "metrics.md", results, counts, n_frames, args)
    print(f"Ghi {out / 'metrics.json'} ({payload['seconds']} giây)")


if __name__ == "__main__":
    main()
