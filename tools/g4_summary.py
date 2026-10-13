"""G4 (require mục 10, chạy ở máy nhà): gộp kết quả các mốc vòng lặp thành checks/g4_overfit_local.json.

  python tools/g4_summary.py --milestone 400=<eval_dir> --milestone 1000=<eval_dir> ... --steps <run>/train_steps.jsonl \
      --overfit_dir data/reltr_overfit --out checks/g4_overfit_local.json
Ghi kèm dấu vân tay code train (third_party/RelTR) và dữ liệu overfit, để 01-checks từ chối bằng chứng cũ khi code/dữ liệu đổi.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD, MAX_ITERS = 90.0, 15000
CODE_GLOBS = ("main.py", "engine.py", "models/*.py", "datasets/*.py", "util/*.py")


def code_fingerprint(reltr_dir):
    h = hashlib.sha256()
    for g in CODE_GLOBS:
        for p in sorted(Path(reltr_dir).glob(g)):
            h.update(p.relative_to(reltr_dir).as_posix().encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))  # bỏ khác biệt CRLF/LF giữa Windows và Kaggle
    return h.hexdigest()


def data_fingerprint(overfit_dir):
    h = hashlib.sha256()
    for name in ("train.json", "rel.json"):
        h.update((Path(overfit_dir) / name).read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", action="append", required=True, help="N=thư mục eval của ag_eval")
    ap.add_argument("--steps", required=True)
    ap.add_argument("--overfit_dir", default=str(ROOT / "data" / "reltr_overfit"))
    ap.add_argument("--reltr_dir", default=str(ROOT / "third_party" / "RelTR"))
    ap.add_argument("--out", default=str(ROOT / "checks" / "g4_overfit_local.json"))
    ap.add_argument("--note", default="", help="ghi chú (ví dụ: code đổi giữa các mốc)")
    a = ap.parse_args()
    traj, traj50 = {}, {}
    for m in a.milestone:
        n, d = m.split("=", 1)
        met = json.load(open(Path(d) / "metrics.json", encoding="utf-8"))["metrics"]["full"]["no@0.5"]
        traj[int(n)], traj50[int(n)] = round(met["R@20"], 2), round(met["R@50"], 2)
    loss = {}
    for line in open(a.steps, encoding="utf-8"):
        s = json.loads(line)
        if s["global_step"] in traj:
            loss[s["global_step"]] = round(s["loss"], 3)
    passed = [n for n in sorted(traj) if traj[n] > THRESHOLD and n <= MAX_ITERS]
    res = {"gate": "G4", "where": "máy nhà (RTX 5060), 1 GPU", "criterion": f"R@20 (No) > {THRESHOLD}% trong <= {MAX_ITERS} vòng lặp",
           "trajectory_R@20_no": {str(k): traj[k] for k in sorted(traj)},
           "trajectory_R@50_no": {str(k): traj50[k] for k in sorted(traj50)},
           "loss": {str(k): v for k, v in sorted(loss.items())},
           "passed_at_iters": passed[0] if passed else None, "pass": bool(passed),
           "config": "--pretrained checkpoint0149.pth --batch_size 2 --lr 1e-4 --lr_backbone 1e-5 --amp, 64 frame reltr_overfit",
           "code_sha256": code_fingerprint(a.reltr_dir), "overfit_data_sha256": data_fingerprint(a.overfit_dir),
           "note": a.note}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
