"""Kiểm thử CPU cho phần lưu/resume (skill.md Bước 0). Chạy: python tests/test_resume_cpu.py"""
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "RelTR"))
from util import checkpoint as ck  # noqa: E402
from util.sampler import ResumableDistributedSampler  # noqa: E402


def test_sampler_cover_and_disjoint():
    n, bs = 103, 2
    for epoch in (0, 1):
        parts = []
        for r in range(2):
            s = ResumableDistributedSampler(n, 2, r, seed=42, batch_size=bs)
            s.set_epoch(epoch)
            parts.append(list(s))
        assert len(parts[0]) == len(parts[1]) == 52
        assert set(parts[0]) | set(parts[1]) == set(range(n)), "hai rank phải phủ đủ dữ liệu"
        # chỉ phần đệm (1 chỉ số) được phép trùng
        assert len(set(parts[0]) & set(parts[1])) <= 1


def test_sampler_skip_matches_uninterrupted():
    n, bs = 101, 2
    for r in range(2):
        full = ResumableDistributedSampler(n, 2, r, seed=42, batch_size=bs)
        full.set_epoch(3)
        ref = list(full)
        part = ResumableDistributedSampler(n, 2, r, seed=42, batch_size=bs)
        part.set_epoch(3, start_iter=7)
        assert list(part) == ref[7 * bs:], "bỏ qua start_iter*batch phải cho đúng phần còn lại"
        assert len(part) == len(ref) - 7 * bs


def test_sampler_epochs_differ():
    s = ResumableDistributedSampler(50, 1, 0, seed=42, batch_size=1)
    assert s.indices_for_epoch(0) != s.indices_for_epoch(1)


def _state(step):
    return {"model": {"w": torch.full((4,), float(step))}, "optimizer": {}, "lr_scheduler": {}, "scaler": None,
            "epoch": 0, "iter_in_epoch": step, "global_step": step, "epoch_complete": False,
            "sampler": {}, "rng": ck.capture_rng(), "meta": {}, "args": {}}


def test_atomic_save_and_fallback():
    d = Path(tempfile.mkdtemp())
    try:
        ckd = d / "runs" / "A1" / "ckpt"
        ck.save_last(ckd, _state(10))
        ck.save_last(ckd, _state(20))
        assert (ckd / "last_prev.pth").exists()
        p, c, gs = ck.select_checkpoint([d])
        assert gs == 20 and p.name == "last.pth"
        # G6 (c): cắt cụt last.pth -> phải tự dùng last_prev.pth
        raw = (ckd / "last.pth").read_bytes()
        (ckd / "last.pth").write_bytes(raw[: len(raw) // 2])
        p, c, gs = ck.select_checkpoint([d])
        assert gs == 10 and p.name == "last_prev.pth", (p, gs)
        # giả lập bị ngắt giữa lúc ghi: còn file .tmp dở dang, last.pth cũ vẫn đọc được
        (ckd / "last.pth.tmp").write_bytes(b"garbage")
        assert ck.try_load(ckd / "last_prev.pth") is not None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_select_max_global_step_across_roots():
    d = Path(tempfile.mkdtemp())
    try:
        ck.save_last(d / "a" / "runs" / "A1" / "ckpt", _state(30))
        ck.save_last(d / "b" / "runs" / "A1" / "ckpt", _state(50))
        ck.save_last(d / "b" / "runs" / "A1" / "ckpt", _state(40))  # last=40, last_prev=50
        p, _, gs = ck.select_checkpoint([d / "a", d / "b"])
        assert gs == 50 and p.name == "last_prev.pth"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_state_dict_sha_stable():
    sd = {"a": torch.arange(5, dtype=torch.float32), "b": torch.ones(2, 2)}
    assert ck.state_dict_sha256(sd) == ck.state_dict_sha256({k: v.clone() for k, v in sd.items()})
    sd2 = dict(sd, b=torch.zeros(2, 2))
    assert ck.state_dict_sha256(sd) != ck.state_dict_sha256(sd2)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} test qua")
