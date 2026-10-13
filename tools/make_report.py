"""FR-7: gộp kết quả thành runs/A1/report.md (máy nhà).

  python tools/make_report.py --run runs/A1 --ann_dir data/ag/annotations --img_folder data/frames
Cần: runs/A1/eval/metrics.json (+ per_frame_full.json), runs/A1/pred_test.pt, A1.yaml, STATUS.json, sessions.csv.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import ag_common as C  # noqa: E402
import ag_eval as E  # noqa: E402
import ag_pred as P  # noqa: E402

# require 11.2 — số từ paper (SGDet, R@10/20/50)
EXTERNAL = [("RelTR (do OED chạy)", "19.7 / 23.4 / 25.9", "20.9 / 24.6 / 28.2", "OED CVPR 2024, Bảng 1", "AG-full"),
            ("STTran", "25.2 / 34.1 / 37.0", "24.6 / 36.2 / 48.8", "OED, Bảng 1", "AG-filtersmall"),
            ("TEMPURA", "28.1 / 33.4 / 34.9", "29.8 / 38.1 / 46.4", "OED, Bảng 1", "AG-filtersmall"),
            ("OED, chỉ không gian", "31.5 / 37.7 / 43.7", "33.4 / 41.6 / 49.0", "OED, Bảng 2", "AG-full"),
            ("OED đầy đủ", "33.5 / 40.9 / 48.9", "35.3 / 44.0 / 51.8", "OED, Bảng 1", "AG-full")]


def ap50_per_class(pred, ann_dir):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gts = C.gt_for_split(ann_dir, "full", "test")
    frames = sorted(gts)
    images, anns, dets = [], [], []
    aid = 1
    for i, f in enumerate(frames, 1):
        g = gts[f]
        images.append({"id": i})
        for b, c in zip(g["boxes"], g["classes"]):
            anns.append({"id": aid, "image_id": i, "category_id": int(c), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         "area": float((b[2] - b[0]) * (b[3] - b[1])), "iscrowd": 0})
            aid += 1
        ent = pred[f]["ent"].float().numpy()
        for x0, y0, x1, y1, lab, sc in ent:
            if 1 <= int(lab) <= 36:
                dets.append({"image_id": i, "category_id": int(lab), "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                             "score": float(sc)})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": k, "name": C.OBJ_CLASSES[k]} for k in range(1, 37)]}
    gt.createIndex()
    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    prec = ev.eval["precision"][0, :, :, 0, 2]  # IoU 0,5; mọi recall; mọi lớp; area all; maxDets 100
    per = {}
    for ki, cid in enumerate(ev.params.catIds):
        p = prec[:, ki]
        per[C.OBJ_CLASSES[cid]] = float(100 * p[p > -1].mean()) if (p > -1).any() else float("nan")
    return float(100 * ev.stats[1]), per


def draw_example(img_folder, f, gt, trip, out_path, topn=5):
    img = Image.open(Path(img_folder) / C.jpg_name(f)).convert("RGB")
    s = 2 if img.width < 600 else 1
    img = img.resize((img.width * s, img.height * s))
    dr = ImageDraw.Draw(img)
    for b in gt["boxes"]:
        dr.rectangle([v * s for v in b], outline=(0, 200, 0), width=2)
    e = E.adapt(trip)
    pr, ps = E.select_pred_rels("with", e["pred_rel_inds"], e["rel_scores"], e["row_group"], e["obj_scores"])
    sc = e["obj_scores"][pr[:, 0]] * e["obj_scores"][pr[:, 1]] * ps
    lines = []
    for j in np.argsort(-sc)[:topn]:
        a, b_, r = pr[j]
        for idx in (a, b_):
            dr.rectangle([v * s for v in e["pred_boxes"][idx]], outline=(230, 30, 30), width=1)
        lines.append(f"{C.OBJ_CLASSES[e['pred_classes'][a]]} -{C.REL_CLASSES[r]}-> {C.OBJ_CLASSES[e['pred_classes'][b_]]} {sc[j]:.2f}")
    for i, t in enumerate(lines):
        dr.text((4, 4 + 12 * i), t, fill=(255, 255, 0))
    img.save(out_path)
    return lines


def interpret(full):
    r20 = full["with@0.5"]["R@20"]
    r20_03 = full.get("with@0.3", {}).get("R@20")
    rules = []
    if r20 < 15:
        rules.append("R@20 (With) < 15 → coi là có lỗi: kiểm lại G2, G3, đường loss, tọa độ box trước khi kết luận.")
    elif r20 >= 30:
        rules.append("R@20 (With) ≥ 30 → RelTR dùng được → đi tiếp TRR.")
    elif 18 <= r20 <= 28:
        rules.append("R@20 (With) trong 18–28 → khớp mốc OED; pipeline đúng; RelTR nguyên bản yếu như dự đoán → A2.")
    else:
        rules.append(f"R@20 (With) = {r20:.2f} nằm ngoài các khoảng của mục 12 (15–18 hoặc 28–30): ghi nhận, người đọc quyết định.")
    rules.append(f"Cần người đọc đánh giá (mục 12 không đặt ngưỡng số): R@20 With IoU 0,3 = {r20_03:.2f} so với IoU 0,5 = {r20:.2f} "
                 "(\"tăng mạnh\"?); AP50 (\"thấp\"?); recall contacting (\"thấp\"?).")
    return rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--img_folder", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_meta", default=str(TOOLS.parent / "data" / "reltr" / "meta.json"),
                    help="meta.json của convert_ag (số frame train thật; chỉ dùng khi sha256 train.json khớp A1.yaml)")
    args = ap.parse_args()
    run = Path(args.run)
    m = json.load(open(run / "eval" / "metrics.json", encoding="utf-8"))
    pred_all = torch.load(run / "pred_test.pt", map_location="cpu", weights_only=False)
    pred, pmeta = pred_all["frames"], pred_all.get("meta", {})
    import yaml
    cfg = yaml.safe_load(open(args.config or (run / "A1.yaml"), encoding="utf-8"))
    st = json.loads((run / "STATUS.json").read_text(encoding="utf-8")) if (run / "STATUS.json").exists() else {}
    sessions = list(csv.DictReader(open(run / "sessions.csv", encoding="utf-8"))) if (run / "sessions.csv").exists() else []
    g8_path = next((p for p in (run / "checks" / "g8_local_vs_kaggle.json", TOOLS.parent / "checks" / "g8_local_vs_kaggle.json")
                    if p.exists()), None)  # require 13: bằng chứng G8 nằm ở checks/ của dự án
    g8 = json.load(open(g8_path, encoding="utf-8")) if g8_path else None
    g8_pass = bool(g8 and (g8.get("pass") or g8.get("compare", {}).get("pass")))
    data_meta = Path(args.data_meta)
    train_meta = json.load(open(data_meta, encoding="utf-8"))["splits"].get("train") if data_meta.exists() else None
    if train_meta and train_meta.get("sha256") != cfg.get("train_json_sha256"):
        train_meta = None  # meta.json không ứng với train.json đã train: không dùng số của nó
    dev_curve = []
    if (run / "log.txt").exists():
        for line in (run / "log.txt").read_text(encoding="utf-8").splitlines():
            d = json.loads(line) if line.startswith("{") else {}
            if "dev_R@20_with" in d:
                dev_curve.append(d["dev_R@20_with"])

    ap50, per_cls = ap50_per_class(pred, args.ann_dir)
    full = m["metrics"]["full"]
    L = ["# Baseline A1 — RelTR fine-tune trên Action Genome (train Kaggle 2× T4, test máy nhà)", ""]
    concl = run / "conclusion.md"
    L += ["## Kết luận", "", concl.read_text(encoding="utf-8").strip() if concl.exists() else
          "_Viết 2–3 câu sau khi đọc các bảng dưới (con số chính, khớp mốc ngoài hay không, hướng tiếp theo theo mục 12)._", ""]
    L += ["Diễn giải theo require mục 12 (tự động cho luật có ngưỡng số):", ""] + [f"- {r}" for r in interpret(full)] + [""]
    for variant, title in (("full", "bảng chính, AG-full"), ("filtersmall", "bảng phụ, AG-filtersmall")):
        if variant not in m["metrics"]:
            continue
        L += [f"## Kết quả SGDet ({title}, {m['n_frames'][variant]} frame)", "",
              "| Chế độ | IoU | R@10 | R@20 | R@50 | mR@10 | mR@20 | mR@50 |", "|---|---|---|---|---|---|---|---|"]
        for key, d in m["metrics"][variant].items():
            mode, iou = key.split("@")
            L.append(f"| {mode} | {iou} | " + " | ".join(f"{d[f'R@{k}']:.1f}" for k in (10, 20, 50)) + " | "
                     + " | ".join(f"{d[f'mR@{k}']:.1f}" for k in (10, 20, 50)) + " |")
        L.append("")
    L += ["## So với mốc ngoài (số từ paper, require 11.2)", "", "| Mô hình | With R@10/20/50 | No R@10/20/50 | Nguồn | So với bảng |", "|---|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in EXTERNAL]
    ours = lambda v, mode: " / ".join(f"{m['metrics'][v][f'{mode}@0.5'][f'R@{k}']:.1f}" for k in (10, 20, 50))
    L += [f"| **A1 (của mình)** | {ours('full', 'with')} | {ours('full', 'no')} | máy nhà | AG-full |"]
    if "filtersmall" in m["metrics"]:
        L += [f"| **A1 (của mình)** | {ours('filtersmall', 'with')} | {ours('filtersmall', 'no')} | máy nhà | AG-filtersmall |"]
    L += ["", "## Chẩn đoán", "", f"- AP50 (entity decoder, AG-full test): **{ap50:.1f}**"]
    srt = sorted(((v, k) for k, v in per_cls.items() if v == v), reverse=True)
    L += ["- AP50 10 lớp tốt nhất: " + ", ".join(f"{k} {v:.1f}" for v, k in srt[:10]),
          "- AP50 10 lớp kém nhất: " + ", ".join(f"{k} {v:.1f}" for v, k in srt[-10:])]
    w5, w3 = full["with@0.5"], full.get("with@0.3", {})
    L += [f"- R@10/20/50 (With) IoU 0,5: {w5['R@10']:.1f} / {w5['R@20']:.1f} / {w5['R@50']:.1f}; IoU 0,3: "
          + (" / ".join(f"{w3[f'R@{k}']:.1f}" for k in (10, 20, 50)) if w3 else "chưa chấm")]
    L += ["- Recall theo nhóm (With, IoU 0,5, R@20): " + ", ".join(f"{g} {v:.1f}" for g, v in w5["per_group@20"].items())]
    cnt = m["gt_triplet_counts"]["full"]
    rare = sorted(C.CON, key=lambda i: cnt[C.REL_CLASSES[i]])[:5]
    L += ["- 5 lớp contacting hiếm nhất (With, R@20): " + ", ".join(
        f"{C.REL_CLASSES[i]} (n={cnt[C.REL_CLASSES[i]]}) {w5['per_class@20'][C.REL_CLASSES[i]]:.1f}" for i in rare), ""]
    groups = (("attention", C.ATT), ("spatial", C.SPA), ("contacting", C.CON))
    L += ["### Recall theo lớp quan hệ (AG-full, R@20)", "",
          "| Nhóm | Lớp | Số bộ ba nhãn thật | With IoU 0,5 | With IoU 0,3 | No IoU 0,5 |", "|---|---|---|---|---|---|"]
    for g, ids in groups:
        for i in sorted(ids, key=lambda i: -cnt[C.REL_CLASSES[i]]):
            r = C.REL_CLASSES[i]
            L.append(f"| {g} | {r} | {cnt[r]} | {w5['per_class@20'][r]:.1f} | "
                     f"{w3['per_class@20'][r]:.1f} | {full['no@0.5']['per_class@20'][r]:.1f} |" if w3 else
                     f"| {g} | {r} | {cnt[r]} | {w5['per_class@20'][r]:.1f} | – | {full['no@0.5']['per_class@20'][r]:.1f} |")
    L.append("")

    pf = json.load(open(run / "eval" / "per_frame_full.json", encoding="utf-8"))
    cand = sorted((v["R@20_with"], v["n_gt"], f) for f, v in pf.items() if v["n_gt"] >= 3)

    def one_per_video(seq, n=5):  # mỗi video tối đa một ví dụ, để 10 ví dụ không dồn vào vài video
        out, seen = [], set()
        for x in seq:
            if x[2].split("/")[0] not in seen:
                seen.add(x[2].split("/")[0])
                out.append(x)
            if len(out) == n:
                break
        return out

    picks = [("tốt", x) for x in one_per_video(cand[::-1])] + [("tệ", x) for x in one_per_video(cand)]
    gts = C.gt_for_split(args.ann_dir, "full", "test", {p[1][2] for p in picks})
    img_dir = run / "report_img"
    img_dir.mkdir(exist_ok=True)
    L += ["## 10 ví dụ minh họa (xanh: box nhãn thật; đỏ: 5 bộ ba điểm cao nhất, With Constraint)", ""]
    for kind, (r, n, f) in picks:
        name = f"{kind}_{f.replace('/', '__')}.jpg"
        lines = draw_example(args.img_folder, f, gts[f], pred[f]["trip"].float().numpy(), img_dir / name)
        L += [f"- **{kind}** `{f}`: R@20 = {100 * r:.0f}%, {n} bộ ba nhãn thật — ![{f}](report_img/{name})"]
    L += ["", "## Cấu hình và các điểm lệch so với chuẩn", "",
          f"- k = {cfg.get('k')} frame/video; backbone: {'mở (lr 1e-5)' if cfg.get('lr_backbone') else 'đóng băng'}; "
          f"batch 2 × 2 GPU; AMP FP16; {cfg.get('epochs')} epoch, lr_drop {cfg.get('lr_drop')}; τ_open = {cfg.get('tau_open')}, "
          f"τ_frozen = {cfg.get('tau_frozen')} s/frame; GPU util {cfg.get('gpu_util')}%",
          (f"- Frame train thật: {train_meta['frames']} frame từ {train_meta['videos']} video (k ≤ 8 frame cách đều mỗi video; "
           f"5% video train tách làm dev), {train_meta['triplets']} bộ ba" if train_meta else "- Frame train thật: không đọc được meta.json"),
          f"- Thời gian train cộng dồn: {st.get('train_seconds', 0) / 3600:.2f} giờ; global_step {st.get('global_step')}; "
          f"số phiên train: {len(sessions)}; số lần resume: {max(len(sessions) - 1, 0)}",
          "- Dev300 R@20 (With) theo epoch (G7, chỉ theo dõi, không chọn checkpoint): "
          + (" / ".join(f"{v:.1f}" for v in dev_curve) if dev_curve else "không có"),
          "- Resume không bảo đảm augmentation (cỡ ảnh, lật) giống hệt chạy liền một mạch (require 9.7).",
          "- Adapter A1: mỗi query là một cặp người–vật, nên dễ hơn With Constraint thật một chút (nhiều query có thể cùng một cặp).",
          "- Semi Constraint không so được với Semi của STTran: STTran chấm spatial/contacting bằng sigmoid (từng lớp độc lập), "
          "adapter A1 chuẩn hóa softmax trong từng nhóm nên hiếm lớp nào vượt 0,9 → Semi thấp hơn With.",
          "- Một lần chạy, seed 42, không có khoảng tin cậy.",
          f"- Test chạy ở máy nhà ({pmeta.get('device')}, torch {pmeta.get('torch')}, AMP {pmeta.get('amp_fp16')}); "
          f"G8: {'QUA' if g8_pass else 'chưa qua hoặc thiếu'} (`checks/g8_local_vs_kaggle.json`)", ""]
    if sessions:
        L += ["| Phiên | T0 | Giờ | Step đầu → cuối | Trạng thái |", "|---|---|---|---|---|"]
        L += [f"| {s['session']} | {s['t0']} | {s['hours']} | {s['start_step']} → {s['end_step']} | {s['state']} |" for s in sessions]
        L.append("")
    env = sorted(run.glob("env_session*.txt"))
    L += ["## Tái lập", "", f"- Checkpoint test: `{pmeta.get('ckpt')}` sha256 `{pmeta.get('ckpt_sha256')}`",
          f"- A1.yaml: train_json_sha256 `{cfg.get('train_json_sha256')}`"]
    if env:
        L += ["- Môi trường Kaggle (phiên đầu):", "```", env[0].read_text(encoding="utf-8").split("# pip freeze")[0].strip(), "```"]
    (run / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Ghi {run / 'report.md'}")


if __name__ == "__main__":
    main()
