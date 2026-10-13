"""Chẩn đoán SAU TEST cho A1 (khám phá, không thuộc giao thức đã chốt ở require mục 11–12; không đặt ngưỡng mới).

Chỉ đọc lại pred_test.pt (MR-2: không chạy lại dự đoán). Trả lời Q2/Q3 chi tiết hơn bằng cách tách mỗi bộ ba
nhãn thật (AG-full, IoU 0,5) vào đúng một nhóm theo khâu đầu tiên bị hỏng:
  1. loc     — không query nào có box người và box vật cùng IoU >= 0,5 (lỗi phát hiện / định vị)
  2. cls     — có query định vị đúng cặp, nhưng không query nào trong số đó đoán đúng lớp vật
  3. pred    — định vị + lớp vật đúng, nhưng không query nào có quan hệ đúng là argmax trong nhóm (With)
  4. rank    — có bộ ba đúng hoàn toàn nhưng xếp hạng ngoài top-20 (With)
  5. hit@20  — tìm thấy trong top-20 (With) — phải khớp R@20 chính thức (kiểm chéo)
Thêm: tỉ lệ ô trong top-20 (With) bị chiếm bởi bộ ba trùng (cùng quan hệ, cùng lớp vật, box người và box vật IoU >= 0,7
với một bộ ba xếp trên).

  python tools/diag_a1.py --pred runs/A1/pred_test.pt --ann_dir data/ag/annotations --out runs/A1/eval/diag_a1.json
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import ag_common as C  # noqa: E402
import ag_eval as E  # noqa: E402
import ag_pred as P  # noqa: E402

K = 20
STAGES = ["loc", "cls", "pred", "rank", "hit"]
GROUP_OF = np.zeros(len(C.REL_CLASSES), dtype=int)
for _g, _s in enumerate(P.GROUP_SLICES):
    GROUP_OF[_s] = _g


def diag_frame(args):
    gt, trip = args
    trip = np.asarray(trip, dtype=float)
    sub, obj = trip[:, P.C_SUB], trip[:, P.C_OBJ]
    pp, ol, os_ = trip[:, P.C_PP], trip[:, P.C_OL].round().astype(int), trip[:, P.C_OS]
    rel = trip[:, P.C_REL]
    q = trip.shape[0]
    # With Constraint qua adapter: mỗi query 3 bộ ba (một mỗi nhóm), điểm = p(người) * p(vật) * max p trong nhóm
    pred_lab = np.stack([rel[:, s].argmax(1) + s.start for s in P.GROUP_SLICES], 1)          # (Q, 3)
    score = (pp * os_)[:, None] * np.stack([rel[:, s].max(1) for s in P.GROUP_SLICES], 1)   # (Q, 3)
    order = score.ravel().argsort()[::-1]
    rank = np.empty(3 * q, dtype=int)
    rank[order] = np.arange(3 * q)
    rank = rank.reshape(q, 3)

    sub_ok = E.bbox_overlaps(gt["boxes"][:1], sub)[0] >= 0.5                                  # (Q,)
    obj_iou = E.bbox_overlaps(gt["boxes"], obj)                                                # (N, Q)
    stages = np.zeros((len(STAGES), 3), dtype=int)                                             # stage x nhóm
    for a, b, r in gt["relations"]:
        j = b if a == 0 else a                                                                 # spatial ghi <vật, người>
        g = GROUP_OF[r]
        loc = sub_ok & (obj_iou[j] >= 0.5)
        cls = loc & (ol == gt["classes"][j])
        prd = cls & (pred_lab[:, g] == r)
        hit = prd & (rank[:, g] < K)
        st = 4 if hit.any() else 3 if prd.any() else 2 if cls.any() else 1 if loc.any() else 0
        stages[st, g] += 1
    n_gt = len(gt["relations"])
    hits = stages[4].sum()

    # ô trùng trong top-K (With)
    top = order[:K]
    tq, tg = top // 3, top % 3
    dup = 0
    for i in range(1, len(top)):
        same = (pred_lab[tq[:i], tg[:i]] == pred_lab[tq[i], tg[i]]) & (ol[tq[:i]] == ol[tq[i]])
        if same.any():
            prev = tq[:i][same]
            si = E.bbox_overlaps(sub[tq[i]][None], sub[prev])[0]
            oi = E.bbox_overlaps(obj[tq[i]][None], obj[prev])[0]
            dup += bool(((si >= 0.7) & (oi >= 0.7)).any())
    return stages, hits / n_gt, dup / len(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    t0 = time.time()
    pred = E.load_pred(args.pred)
    gts = C.gt_for_split(args.ann_dir, "full", "test")
    frames = sorted(gts)
    with Pool(args.workers) as pool:
        res = pool.map(diag_frame, [(gts[f], pred[f]["trip"].float().numpy()) for f in frames], chunksize=64)
    stages = sum(r[0] for r in res)
    tot_g = stages.sum(0)
    out = {
        "note": "Chẩn đoán sau test, khám phá; không thuộc giao thức đã chốt (require 11–12). AG-full, IoU 0,5, With, K=20.",
        "n_frames": len(frames), "n_gt_triplets": int(tot_g.sum()),
        "check_R@20_with_from_diag": round(100 * float(np.mean([r[1] for r in res])), 2),
        "stage_share_all": {s: round(100 * float(stages[i].sum() / tot_g.sum()), 2) for i, s in enumerate(STAGES)},
        "stage_share_by_group": {g: {s: round(100 * float(stages[i, k] / tot_g[k]), 2) for i, s in enumerate(STAGES)}
                                 for k, g in enumerate(["attention", "spatial", "contacting"])},
        "dup_share_top20": round(100 * float(np.mean([r[2] for r in res])), 2),
        "seconds": round(time.time() - t0, 1), "time": C.now_str(),
    }
    C.dump_json(out, args.out, indent=1)
    print(out)


if __name__ == "__main__":
    main()
