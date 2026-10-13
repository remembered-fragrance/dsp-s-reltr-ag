"""Cài demo trên máy mới sau khi clone repo (Windows, Linux, macOS).

  python deploy/setup_demo.py            # tự chọn torch GPU (CUDA 12.8) nếu có nvidia-smi, ngược lại CPU
  python deploy/setup_demo.py --cpu      # ép bản CPU
Các bước: tạo .venv -> cài torch + thư viện -> dựng RelTR (tools/setup_reltr.py) -> tải trọng số A1 từ GitHub Release
(cần `gh auth login`; dự phòng: repo model Hugging Face qua DSP_MODEL_REPO + `hf auth login`) -> kiểm sha256.
Cần sẵn: Python 3.10–3.13, git; gh (GitHub CLI) để tải trọng số từ repo private.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
CKPT = ROOT / "runs" / "A1" / "ckpt" / "epoch4_weights.pth"
CKPT_SHA256 = "e6988797121af05090c178dbe85212885e5d15caaf2adc0788921a4206c9362c"
RELEASE_TAG = "weights-a1"
TORCH = ["torch==2.11.0", "torchvision==0.26.0"]
INDEX = {"cuda": "https://download.pytorch.org/whl/cu128", "cpu": "https://download.pytorch.org/whl/cpu"}
GRADIO = "gradio==6.30.0"


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def venv_python(venv):
    p = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return p


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def base_requirements():
    """requirements.txt trừ torch/torchvision và dòng index (đã cài riêng theo GPU/CPU)."""
    out = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        s = line.split(" #", 1)[0].strip()  # bỏ ghi chú cuối dòng
        if not s or s.startswith("#") or s.startswith("-") or s.split("=")[0].strip() in ("torch", "torchvision"):
            continue
        out.append(s)
    return out


def origin_repo():
    url = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"], capture_output=True, text=True).stdout.strip()
    url = url.removesuffix(".git")
    return "/".join(url.replace(":", "/").split("/")[-2:]) if url else None


def fetch_weights():
    if CKPT.exists() and sha256(CKPT) == CKPT_SHA256:
        print(f"trọng số đã có và đúng sha256: {CKPT}")
        return
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    repo = origin_repo()
    if shutil.which("gh") and repo:
        run(["gh", "release", "download", RELEASE_TAG, "-R", repo, "-p", CKPT.name, "-D", CKPT.parent, "--clobber"])
    elif os.environ.get("DSP_MODEL_REPO"):
        from huggingface_hub import hf_hub_download  # có trong requirements
        p = hf_hub_download(os.environ["DSP_MODEL_REPO"], CKPT.name, token=os.environ.get("HF_TOKEN"))
        shutil.copy2(p, CKPT)
    else:
        sys.exit("không tải được trọng số: cài GitHub CLI và chạy `gh auth login` (hoặc đặt DSP_MODEL_REPO và đăng nhập `hf auth login`)")
    got = sha256(CKPT)
    if got != CKPT_SHA256:
        CKPT.unlink()
        sys.exit(f"sha256 trọng số sai ({got}); đã xóa file, chạy lại")
    print(f"trọng số OK (sha256 khớp): {CKPT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="cài torch bản CPU dù máy có GPU NVIDIA")
    ap.add_argument("--venv", default=str(VENV), help="thư mục môi trường ảo (mặc định .venv trong repo)")
    ap.add_argument("--skip_install", action="store_true", help="bỏ qua bước pip (đã cài sẵn)")
    a = ap.parse_args()
    if sys.version_info < (3, 10):
        sys.exit("cần Python >= 3.10")
    venv = Path(a.venv)
    py = venv_python(venv)
    if not py.exists():
        run([sys.executable, "-m", "venv", venv])
    if not a.skip_install:
        kind = "cpu" if a.cpu or not shutil.which("nvidia-smi") else "cuda"
        print(f"cài PyTorch bản {kind.upper()}")
        run([py, "-m", "pip", "install", "--upgrade", "pip"])
        run([py, "-m", "pip", "install", *TORCH, "--index-url", INDEX[kind]])
        run([py, "-m", "pip", "install", *base_requirements(), GRADIO])
    run([py, ROOT / "tools" / "setup_reltr.py"])
    fetch_weights()
    run([py, "-c", "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available())"])
    act = os.path.relpath(py, ROOT) if py.resolve().is_relative_to(ROOT) else str(py)  # --venv có thể ở ngoài repo
    print("\nXong. Chạy demo:")
    print(f"  {act} demo/app.py                       # chỉ máy này: http://127.0.0.1:7860")
    if os.name == "nt":
        print("  powershell -ExecutionPolicy Bypass -File deploy/serve_local.ps1   # mở link cho máy khác (có mật khẩu)")
    else:
        print("  bash deploy/serve_local.sh                       # mở link cho máy khác (có mật khẩu)")


if __name__ == "__main__":
    main()
