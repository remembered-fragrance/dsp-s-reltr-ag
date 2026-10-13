"""G5 (require mục 8): đo tốc độ và chốt k, backbone, num_workers.

Đo (sau một lần `torchrun ... main.py --max_iters 220 --print_freq 1`):
  python tools/g5_speed.py measure --run <out_dir> --label open --world 2 --batch 2 --gpu_util <csv> --out checks/g5_open.json
Quyết định (sau khi có các file đo):
  python tools/g5_speed.py decide --open checks/g5_open.json [--frozen checks/g5_frozen.json] [--extra checks/g5_open_w3.json]
      --n_videos 7205 --config configs/A1.yaml --out checks/g5_speed.txt
Ghi sha256 của train.json sinh với k đã chốt:
  python tools/g5_speed.py sha --config configs/A1.yaml --train_json <ann_path>/train.json
"""
import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import yaml

WARMUP, MEASURE, EPOCHS, BUDGET_S, K_MAX = 20, 200, 4, 36000, 8


def measure(args):
    steps = {}
    for line in open(Path(args.run) / "train_steps.jsonl", encoding="utf-8"):
        d = json.loads(line)
        steps[d["global_step"]] = d["time"]
    a, b = WARMUP, WARMUP + MEASURE
    assert a in steps and b in steps, f"thiếu step {a} hoặc {b} trong train_steps.jsonl (cần --print_freq 1, --max_iters >= {b})"
    tau = (steps[b] - steps[a]) / (MEASURE * args.world * args.batch)
    util = None
    if args.gpu_util and Path(args.gpu_util).exists():
        vals = []
        for line in open(args.gpu_util, encoding="utf-8"):
            parts = [p.strip() for p in line.split(",")]
            try:
                ts = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                if steps[a] <= ts <= steps[b]:
                    vals.append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
        util = round(sum(vals) / len(vals), 1) if vals else None
    res = {"label": args.label, "tau_s_per_frame": round(tau, 5), "gpu_util_mean": util, "world": args.world,
           "batch_per_gpu": args.batch, "steps": [a, b], "seconds": round(steps[b] - steps[a], 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))


def k_for(tau, n_videos):
    return math.floor(BUDGET_S / (tau * EPOCHS * n_videos))


def decide(args):
    op = json.load(open(args.open))
    lines = [f"N (video train sau khi tách dev) = {args.n_videos}",
             f"tau_open = {op['tau_s_per_frame']} s/frame, GPU util {op['gpu_util_mean']}%"]
    cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    k_open = k_for(op["tau_s_per_frame"], args.n_videos)
    lines.append(f"Quy tắc 1: k_open = floor(36000 / (tau_open * 4 * N)) = {k_open}")
    chosen = op
    if k_open >= 6:
        k, lr_backbone = min(k_open, K_MAX), 1e-5
        lines.append(f"-> mở backbone (lr 1e-5), k = {k}")
    else:
        assert args.frozen, "k_open < 6: cần đo thêm trường hợp đóng băng backbone (--frozen)"
        fr = json.load(open(args.frozen))
        chosen = fr
        k_fr = k_for(fr["tau_s_per_frame"], args.n_videos)
        k, lr_backbone = min(k_fr, K_MAX), 0.0
        lines.append(f"Quy tắc 2: tau_frozen = {fr['tau_s_per_frame']} -> k = min({k_fr}, 8) = {k}; đóng băng backbone")
    epochs = EPOCHS
    if k < 3:
        k, epochs = 3, 3
        lines.append("Quy tắc 3: k < 3 -> k = 3, epochs = 3 (PHẢI ghi nhật ký thay đổi require mục 17)")
    num_workers = cfg.get("num_workers") or 2
    if chosen.get("gpu_util_mean") is not None and chosen["gpu_util_mean"] < 70:
        lines.append(f"Quy tắc 4: GPU util {chosen['gpu_util_mean']}% < 70% -> dataloader là nút cổ chai")
        if args.extra:
            ex = json.load(open(args.extra))
            if ex["tau_s_per_frame"] < chosen["tau_s_per_frame"]:
                num_workers = 3
                lines.append(f"  num_workers 3 nhanh hơn (tau {ex['tau_s_per_frame']}) -> dùng num_workers 3")
            else:
                lines.append("  num_workers 3 không nhanh hơn -> giữ 2")
        else:
            lines.append("  cần đo thêm một lần với --num_workers 3 (--extra)")
    cfg.update({"k": int(k), "lr_backbone": float(lr_backbone), "num_workers": int(num_workers), "batch_size": 2,
                "epochs": int(epochs), "lr_drop": 3, "tau_open": op["tau_s_per_frame"],
                "tau_frozen": json.load(open(args.frozen))["tau_s_per_frame"] if args.frozen else None,
                "gpu_util": chosen.get("gpu_util_mean"), "decided_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "train_json_sha256": None})
    est_h = chosen["tau_s_per_frame"] * k * args.n_videos * epochs / 3600
    lines.append(f"Thời gian train ước tính: {est_h:.1f} giờ (SR-4: <= 10 giờ)")
    yaml.safe_dump(cfg, open(args.config, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Đã ghi {args.config}. Bước tiếp: sinh lại train.json với k={k} rồi chạy `g5_speed.py sha`.")


def sha(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ag_common as C
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    meta = json.load(open(Path(args.train_json).parent / "meta.json", encoding="utf-8"))
    assert meta["k"] == cfg["k"], f"train.json sinh với k={meta['k']} nhưng A1.yaml chốt k={cfg['k']}"
    cfg["train_json_sha256"] = C.sha256_file(args.train_json)
    yaml.safe_dump(cfg, open(args.config, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    print(f"train_json_sha256 = {cfg['train_json_sha256']} -> {args.config}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--run", required=True)
    m.add_argument("--label", required=True)
    m.add_argument("--world", type=int, default=2)
    m.add_argument("--batch", type=int, default=2)
    m.add_argument("--gpu_util")
    m.add_argument("--out", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--open", required=True)
    d.add_argument("--frozen")
    d.add_argument("--extra")
    d.add_argument("--n_videos", type=int, required=True)
    d.add_argument("--config", required=True)
    d.add_argument("--out", required=True)
    s = sub.add_parser("sha")
    s.add_argument("--config", required=True)
    s.add_argument("--train_json", required=True)
    args = ap.parse_args()
    {"measure": measure, "decide": decide, "sha": sha}[args.cmd](args)


if __name__ == "__main__":
    main()
