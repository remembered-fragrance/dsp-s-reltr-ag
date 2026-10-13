"""Kiểm thử CPU: P2–P4, P6 (đường không AMP), đóng gói dự đoán và adapter đánh giá.
Chạy: python tests/test_model_cpu.py [--ckpt data/ckpt/checkpoint0149.pth]"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RELTR = ROOT / "third_party" / "RelTR"
sys.path.insert(0, str(RELTR))
sys.path.insert(0, str(ROOT / "tools"))

try:  # dataset không dùng trong test này; stub chỉ để import main.py khi máy chưa có pycocotools
    import pycocotools  # noqa: F401
except ImportError:
    pc = types.ModuleType("pycocotools")
    pc.mask = types.ModuleType("pycocotools.mask")
    pc.coco = types.ModuleType("pycocotools.coco")
    pc.coco.COCO = object
    pc.cocoeval = types.ModuleType("pycocotools.cocoeval")
    pc.cocoeval.COCOeval = object
    sys.modules.update({"pycocotools": pc, "pycocotools.mask": pc.mask, "pycocotools.coco": pc.coco,
                        "pycocotools.cocoeval": pc.cocoeval})

from main import get_args_parser, load_pretrained  # noqa: E402
from engine import TrainControl, outputs_to_float  # noqa: E402
from models import build_model  # noqa: E402
from util.misc import nested_tensor_from_tensor_list  # noqa: E402
import ag_eval  # noqa: E402
import ag_pred  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "data" / "ckpt" / "checkpoint0149.pth"))
    a = ap.parse_args()
    torch.manual_seed(0)
    args = argparse.ArgumentParser(parents=[get_args_parser()]).parse_args(
        ["--dataset", "ag", "--device", "cpu", "--lr_backbone", "0"])
    model, criterion, _ = build_model(args)
    assert model.entity_class_embed.weight.shape == (38, 256)
    assert model.sub_class_embed.weight.shape == (38, 256) and model.obj_class_embed.weight.shape == (38, 256)
    assert model.rel_class_embed.layers[-1].weight.shape == (28, 256)
    assert criterion.empty_weight_rel.shape == (28,) and criterion.num_rel_classes == 27
    print("ok P2/P3: shape các đầu phân loại đúng")

    if Path(a.ckpt).exists():
        info = load_pretrained(model, a.ckpt)
        print("ok P4: chỉ bỏ", len(info["dropped"]), "key đầu phân loại")
    else:
        print("bỏ qua P4: không có", a.ckpt)

    assert not any(p.requires_grad for n, p in model.named_parameters() if "backbone" in n), "lr_backbone=0 phải đóng băng backbone"

    imgs = [torch.randn(3, 448, 600), torch.randn(3, 480, 560)]
    targets = []
    for _ in imgs:
        targets.append({"labels": torch.tensor([1, 11, 32]),
                        "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.8], [0.3, 0.6, 0.1, 0.1], [0.6, 0.8, 0.5, 0.3]]),
                        "rel_annotations": torch.tensor([[0, 1, 1], [0, 1, 6], [0, 1, 15], [0, 2, 2], [0, 2, 5]]),
                        "image_id": torch.tensor([1]), "orig_size": torch.tensor([270, 480]), "size": torch.tensor([448, 600])})
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    out = outputs_to_float(model(nested_tensor_from_tensor_list(imgs)))
    losses = criterion(out, targets)
    loss = sum(losses[k] * criterion.weight_dict[k] for k in losses if k in criterion.weight_dict)
    assert torch.isfinite(loss), loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    opt.step()
    print(f"ok P1/P6: forward + backward + bước optimizer, loss={loss.item():.3f}")

    model.eval()
    with torch.no_grad():
        o = model(nested_tensor_from_tensor_list([imgs[0]]))
    packed = ag_pred.pack_outputs(o, [(480, 270)])[0]
    trip, ent = packed["trip"], packed["ent"]
    assert trip.shape == (100, 37) and ent.shape == (100, 6) and trip.dtype == torch.float16
    rel = trip[:, ag_pred.C_REL].float()
    for s in ag_pred.GROUP_SLICES:
        assert torch.allclose(rel[:, s].sum(-1), torch.ones(100), atol=1e-2), "mỗi nhóm quan hệ phải cộng lại bằng 1"
    assert ((trip[:, ag_pred.C_OL] >= 2) & (trip[:, ag_pred.C_OL] <= 36)).all()
    print("ok FR-5: đóng gói dự đoán đúng định dạng")

    gt = {"boxes": np.array([[10, 10, 200, 260], [50, 60, 90, 100.]]), "classes": np.array([1, 11]),
          "relations": np.array([[0, 1, 0], [1, 0, 5], [0, 1, 14]])}
    res, cnt = ag_eval.eval_frame((gt, trip.float().numpy(), ["with", "semi", "no"], [0.5]))
    assert set(res) == {("with", 0.5), ("semi", 0.5), ("no", 0.5)} and cnt.sum() == 3
    print("ok FR-6: adapter + bộ đánh giá chạy trên dự đoán thật của model")

    c = TrainControl(global_step=38, max_iters=40, check_every=20)
    flags = []
    for _ in range(3):
        c.global_step += 1
        flags.append(c.decide())
    assert flags == [TrainControl.CONTINUE, TrainControl.MAX_ITERS, TrainControl.MAX_ITERS], flags
    c2 = TrainControl(global_step=19, deadline_unix=1.0, check_every=20)
    c2.global_step += 1
    assert c2.decide() == TrainControl.DEADLINE
    print("ok P12: quyết định dừng theo --max_iters và hạn giờ")
    print("tất cả test model CPU qua")


if __name__ == "__main__":
    main()
