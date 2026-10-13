"""G0 (require mục 10): chạy checkpoint RelTR VG gốc trên ảnh demo, in 10 bộ ba điểm cao nhất (logic như inference.py
của RelTR, nhưng nạp với weights_only=False và chạy được trên CPU/GPU, không vẽ hình).

  python tools/g0_vg_inference.py --ckpt data/ckpt/checkpoint0149.pth --img third_party/RelTR/demo/vg1.jpg --out checks/g0_vg.txt
"""
import argparse
import re
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

RELTR = Path(__file__).resolve().parents[1] / "third_party" / "RelTR"
sys.path.insert(0, str(RELTR))


def vg_lists():
    src = (RELTR / "inference.py").read_text(encoding="utf-8")
    grab = lambda name: re.findall(r"'([^']*)'", src.split(f"{name} = [", 1)[1].split("]", 1)[0])
    return grab("CLASSES"), grab("REL_CLASSES")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img", default=str(RELTR / "demo" / "vg1.jpg"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out")
    a = ap.parse_args()
    from main import get_args_parser
    from models import build_model
    args = __import__("argparse").ArgumentParser(parents=[get_args_parser()]).parse_args(["--dataset", "vg", "--device", a.device])
    model, _, _ = build_model(args)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.to(a.device).eval()
    classes, rels = vg_lists()
    im = Image.open(a.img).convert("RGB")
    x = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(im)
    o = model(x.unsqueeze(0).to(a.device))
    pr = o["rel_logits"].softmax(-1)[0, :, :-1]
    ps = o["sub_logits"].softmax(-1)[0, :, :-1]
    po = o["obj_logits"].softmax(-1)[0, :, :-1]
    keep = (pr.max(-1).values > 0.3) & (ps.max(-1).values > 0.3) & (po.max(-1).values > 0.3)
    q = torch.nonzero(keep, as_tuple=True)[0]
    order = torch.argsort(-(pr[q].max(-1)[0] * ps[q].max(-1)[0] * po[q].max(-1)[0]))[:10]
    lines = [f"ảnh {a.img}, thiết bị {a.device}, {int(keep.sum())} query vượt ngưỡng 0,3"]
    for i in q[order]:
        lines.append(f"{classes[ps[i].argmax()]} — {rels[pr[i].argmax()]} — {classes[po[i].argmax()]}  "
                     f"({ps[i].max():.2f} × {pr[i].max():.2f} × {po[i].max():.2f})")
    print("\n".join(lines))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
