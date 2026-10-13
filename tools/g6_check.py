"""G6 (require mục 10): tổng hợp các kiểm tra lưu và resume thành checks/g6_resume.json.

  python tools/g6_check.py --ref <run_ref> --res <run_resumed> --resume_step 40 --final_step 60 \
      --deadline_run <run_deadline> --deadline_rc 0 --corrupt_json <restore_state output .json> \
      --readonly_rc 0 [--push ok|fail|unused] --out checks/g6_resume.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "RelTR"))
from util import checkpoint as ck  # noqa: E402


def ids_by_step(run):
    out = {}
    for f in sorted(Path(run).glob("sample_ids_r*.jsonl")):
        r = f.stem.split("_r")[-1]
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            out[(r, d["step"])] = d["ids"]  # resume ghi đè các step đã chạy lại (không nên có)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--resume_step", type=int, default=40)
    ap.add_argument("--final_step", type=int, default=60)
    ap.add_argument("--deadline_run", required=True)
    ap.add_argument("--deadline_rc", type=int, required=True)
    ap.add_argument("--corrupt_json", required=True, help="dòng JSON cuối của restore_state sau khi cắt cụt last.pth")
    ap.add_argument("--readonly_rc", type=int, required=True)
    ap.add_argument("--push", default="unused", choices=["ok", "fail", "unused"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref, res = ids_by_step(a.ref), ids_by_step(a.res)
    steps = range(a.resume_step + 1, a.final_step + 1)
    ranks = sorted({r for r, _ in ref})
    mism = [(r, s) for r in ranks for s in steps if ref.get((r, s)) != res.get((r, s))]
    info = json.load(open(Path(a.res) / "checks" / f"resume_info_step{a.resume_step}.json", encoding="utf-8"))
    st_res = ck.read_status(a.res)
    st_dl = ck.read_status(a.deadline_run)
    corrupt = json.loads(Path(a.corrupt_json).read_text(encoding="utf-8").strip().splitlines()[-1])
    checks = {
        "a_deadline": {"state": st_dl.get("state"), "rc": a.deadline_rc,
                       "pass": st_dl.get("state") == "time_limit" and a.deadline_rc == 0},
        "b_sample_ids": {"ranks": ranks, "steps": [a.resume_step + 1, a.final_step], "mismatch": mism[:10],
                         "pass": not mism and bool(ranks)},
        "b_global_step": {"resumed": info["global_step"], "final": st_res.get("global_step"),
                          "pass": info["global_step"] == a.resume_step and st_res.get("global_step") == a.final_step},
        "b_lr_scale": {"lr_now": info["lr_now"], "lr_saved": info["lr_saved"], "scale_now": info["scale_now"],
                       "scale_saved": info["scale_saved"],
                       "pass": info["lr_now"] == info["lr_saved"] and info["scale_now"] == info["scale_saved"]},
        "b_weights_sha256": {"pass": info["weights_sha256_now"] == info["weights_sha256_saved"]},
        "c_corrupt_fallback": {"chosen": corrupt.get("ckpt"), "pass": str(corrupt.get("ckpt", "")).endswith("last_prev.pth")},
        "d_readonly": {"rc": a.readonly_rc, "pass": a.readonly_rc == 0},
        "e_push": {"result": a.push, "pass": a.push in ("ok", "unused")},
    }
    out = {"checks": checks, "pass": all(c["pass"] for c in checks.values())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, c in checks.items():
        print(f"{k}: {'QUA' if c['pass'] else 'KHÔNG QUA'}")
    print("G6", "QUA" if out["pass"] else "KHÔNG QUA")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
