"""FR-5: dự đoán RelTR trên một split AG ở chế độ test (MR-3: cạnh ngắn 600, cạnh dài <= 1000).

Máy nhà (1 GPU, mặc định):
  python tools/predict_ag.py --ckpt runs/A1/ckpt/epoch4_weights.pth --ann_path data/reltr --img_folder data/frames \
      --split test --out_dir runs/A1/pred_parts
  python tools/predict_ag.py --merge runs/A1/pred_parts --split test --ann_path data/reltr --out runs/A1/pred_test.pt
Tập nhỏ (G4, G8): --frame_list <txt> --out <file.pt> (lưu một file, không chia phần).
Kaggle (dự phòng): torchrun --nproc_per_node=2 tools/predict_ag.py ... (mỗi tiến trình một nửa danh sách, chia chẵn/lẻ).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import ag_common as C  # noqa: E402
import ag_pred as P  # noqa: E402

DEFAULT_RELTR = TOOLS.parent / "third_party" / "RelTR"
SPLIT_FILE = {"test": "test.json", "dev": "val.json", "train": "train.json"}


def import_reltr(reltr_dir):
    sys.path.insert(0, str(Path(reltr_dir).resolve()))
    from main import get_args_parser  # noqa: E402
    from models import build_model  # noqa: E402
    import datasets.transforms as T  # noqa: E402
    from datasets.coco import make_ag_transforms  # noqa: E402
    return get_args_parser, build_model, T, make_ag_transforms


def load_model(ckpt_path, device, reltr_dir=DEFAULT_RELTR):
    """Dựng RelTR cho AG và nạp trọng số (epoch*_weights.pth hoặc last.pth). Trả về (model, transform)."""
    get_args_parser, build_model, _, make_ag_transforms = import_reltr(reltr_dir)
    args = argparse.ArgumentParser(parents=[get_args_parser()]).parse_args(
        ["--dataset", "ag", "--device", str(device), "--lr_backbone", "0"])
    model, _, _ = build_model(args)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=True)
    return model.to(device).eval(), make_ag_transforms("test")


class FrameSet(Dataset):
    def __init__(self, img_folder, frames, transform):
        self.img_folder, self.frames, self.transform = Path(img_folder), frames, transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        fn = self.frames[i]
        img = Image.open(self.img_folder / fn).convert("RGB")
        w, h = img.size
        x, _ = self.transform(img, None)
        return x, (w, h), fn


def _collate(batch):
    return batch[0]


@torch.no_grad()
def run_frames(model, loader, device, amp, deadline=0):
    from util.misc import nested_tensor_from_tensor_list
    out, stopped = {}, False
    for x, (w, h), fn in loader:
        if C.deadline_passed(deadline):
            stopped = True
            break
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            o = model(nested_tensor_from_tensor_list([x.to(device, non_blocking=True)]))
        out[C.png_key(fn)] = P.pack_outputs(o, [(w, h)])[0]
    return out, stopped


def frame_list_for(args):
    js = json.load(open(Path(args.ann_path) / SPLIT_FILE[args.split], encoding="utf-8"))
    frames = [im["file_name"] for im in js["images"]]
    if args.frame_list:
        want = {C.jpg_name(C.png_key(l.strip())) for l in open(args.frame_list, encoding="utf-8") if l.strip()}
        frames = [f for f in frames if f in want]
        assert len(frames) == len(want), f"{len(want) - len(frames)} frame trong --frame_list không có trong {SPLIT_FILE[args.split]}"
    return sorted(frames)


def merge(args):
    parts = sorted(Path(args.merge).glob(f"pred_{args.split}_r*_p*.pt"))
    frames = {}
    for p in parts:
        frames.update(torch.load(p, map_location="cpu", weights_only=False))
    expected = {C.png_key(f) for f in frame_list_for(args)}
    miss = expected - set(frames)
    assert not miss, f"thiếu dự đoán cho {len(miss)} frame, ví dụ {sorted(miss)[:3]} (chạy lại lệnh dự đoán để làm tiếp)"
    meta = json.loads((Path(args.merge) / "pred_meta.json").read_text(encoding="utf-8"))
    meta.update({"merged_parts": len(parts), "n_frames": len(frames), "merged_at": C.now_str()})
    torch.save({"frames": frames, "meta": meta}, args.out)
    C.dump_json(meta, Path(args.out).with_name(Path(args.out).stem + "_meta.json"), indent=1)
    print(f"Gộp {len(parts)} phần, {len(frames)} frame -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reltr_dir", default=str(DEFAULT_RELTR))
    ap.add_argument("--ckpt")
    ap.add_argument("--ann_path", required=True)
    ap.add_argument("--img_folder")
    ap.add_argument("--split", default="test", choices=list(SPLIT_FILE))
    ap.add_argument("--frame_list")
    ap.add_argument("--out_dir", help="lưu theo phần (test)")
    ap.add_argument("--out", help="một file .pt (tập nhỏ) hoặc đích của --merge")
    ap.add_argument("--merge")
    ap.add_argument("--part_size", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--deadline_unix", type=float, default=0)
    ap.add_argument("--max_frames", type=int, default=0, help="chỉ để thử nghiệm")
    args = ap.parse_args()
    if args.merge:
        return merge(args)
    assert args.ckpt and args.img_folder and (args.out_dir or args.out)

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if args.device == "cuda" and world > 1:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device(args.device if args.device != "cuda" else f"cuda:{torch.cuda.current_device()}")
    amp = device.type == "cuda" and not args.no_amp
    model, transform = load_model(args.ckpt, device, args.reltr_dir)

    frames = frame_list_for(args)
    if args.max_frames:
        frames = frames[: args.max_frames]
    mine = frames[rank::world]
    meta = {"ckpt": str(args.ckpt), "ckpt_sha256": C.sha256_file(args.ckpt), "split": args.split,
            "frame_list": args.frame_list, "n_frames_total": len(frames), "world_size": world, "amp_fp16": amp,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__, "test_size": "short 600 / long <= 1000", "topq": P.TOPQ,
            "started": C.now_str(), "format": "ag_pred.py: trip (Q,37) + ent (100,6), float16"}
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    t0 = time.time()

    def loader(fl):
        return DataLoader(FrameSet(args.img_folder, fl, transform), batch_size=1, shuffle=False,
                          num_workers=args.num_workers, collate_fn=_collate, pin_memory=device.type == "cuda")

    if args.out and not args.out_dir:  # tập nhỏ, một file
        preds, stopped = run_frames(model, loader(mine), device, amp, args.deadline_unix)
        assert not stopped, "hết giờ khi đang dự đoán tập nhỏ"
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"frames": preds, "meta": meta}, args.out)
        print(f"{len(preds)} frame -> {args.out} ({time.time() - t0:.0f} giây)")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        C.dump_json(meta, out_dir / "pred_meta.json", indent=1)
    n_parts = (len(mine) + args.part_size - 1) // args.part_size
    for p in range(n_parts):
        f = out_dir / f"pred_{args.split}_r{rank}_p{p:03d}.pt"
        if f.exists():
            continue
        chunk = mine[p * args.part_size:(p + 1) * args.part_size]
        preds, stopped = run_frames(model, loader(chunk), device, amp, args.deadline_unix)
        if stopped:
            print(f"Tới hạn giờ ở phần {p}: phần này chưa lưu, chạy lại lệnh để làm tiếp", flush=True)
            return
        tmp = f.with_suffix(".tmp")
        torch.save(preds, tmp)
        os.replace(tmp, f)
        el = time.time() - t0
        print(f"[r{rank}] phần {p + 1}/{n_parts} xong ({len(preds)} frame), {el / 60:.1f} phút", flush=True)
    print(f"[r{rank}] xong {len(mine)} frame trong {(time.time() - t0) / 60:.1f} phút")


if __name__ == "__main__":
    main()
