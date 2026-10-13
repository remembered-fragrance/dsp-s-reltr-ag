"""FR-10: giao diện demo Gradio.

Máy nhà (mặc định 127.0.0.1, không bật share):
  python demo/app.py --ckpt runs/A1/ckpt/epoch4_weights.pth
Hugging Face Space private (đóng gói bởi space/build_space.py): Space đặt GRADIO_SERVER_NAME; trọng số tải từ repo
model private DSP_MODEL_REPO bằng secret HF_TOKEN và được kiểm sha256 trước khi dùng.
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import demo as D  # noqa: E402

CKPT_FILE = os.environ.get("DSP_CKPT_FILE", "epoch4_weights.pth")
CKPT_SHA256 = os.environ.get("DSP_CKPT_SHA256", "e6988797121af05090c178dbe85212885e5d15caaf2adc0788921a4206c9362c")
MAX_SECONDS = float(os.environ.get("DSP_MAX_SECONDS", "60"))  # giới hạn thời lượng xử lý (CPU của Space chậm)


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def resolve_ckpt(path):
    """Dùng file cục bộ nếu có; nếu không thì tải từ repo model DSP_MODEL_REPO (private, cần HF_TOKEN)."""
    if path and Path(path).exists():
        return str(path)
    repo = os.environ.get("DSP_MODEL_REPO")
    if not repo:
        sys.exit(f"không có {path} và chưa đặt DSP_MODEL_REPO: tải trọng số trước (skill Bước 5a)")
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=repo, filename=CKPT_FILE, token=os.environ.get("HF_TOKEN"))
    got = sha256(p)
    if CKPT_SHA256 and got != CKPT_SHA256:
        sys.exit(f"sha256 của {CKPT_FILE} là {got}, khác giá trị mong đợi {CKPT_SHA256}")
    return p


def build(ckpt):
    import gradio as gr
    import torch

    gpu = torch.cuda.is_available()

    def infer(video, fps, thr, topk, progress=gr.Progress()):
        if not video:
            raise gr.Error("Chưa chọn video")
        out = ROOT / "demo_out" / f"app_{time.strftime('%Y%m%d_%H%M%S')}"
        mp4, gj, graphs = D.run(video, ckpt, out, fps=fps, thr=thr, topk=int(topk), progress=lambda p: progress(p),
                                source=f"video tải lên giao diện Gradio ({Path(video).name})", max_seconds=MAX_SECONDS)
        rows = [[g["time_s"], t["predicate"], t["object"], t["score"]] for g in graphs for t in g["triplets"]]
        return str(mp4), rows, str(gj), (out / "summary.md").read_text(encoding="utf-8")

    with gr.Blocks(title="DSP_S — RelTR trên Action Genome") as app:
        gr.Markdown("## Video → đồ thị cảnh theo từng frame (RelTR baseline A1)\n"
                    f"Demo minh họa; không dùng số liệu ở đây trong báo cáo. Chạy trên **{'GPU' if gpu else 'CPU'}**; "
                    f"chỉ xử lý {MAX_SECONDS:.0f} giây đầu của video"
                    + ("" if gpu else " (CPU: khoảng 1–2 giây cho mỗi frame)") + ".")
        with gr.Row():
            with gr.Column():
                vin = gr.Video(label="Video đầu vào")
                fps = gr.Slider(0.5, 5, value=2 if gpu else 1, step=0.5, label="Số frame mỗi giây")
                thr = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="Ngưỡng điểm bộ ba")
                topk = gr.Slider(1, 20, value=10, step=1, label="Tối đa bộ ba mỗi frame")
                btn = gr.Button("Chạy", variant="primary")
            with gr.Column():
                vout = gr.Video(label="Video đã chú thích")
                fout = gr.File(label="graphs.json")
        table = gr.Dataframe(headers=["Thời điểm (s)", "Quan hệ", "Vật", "Điểm"], label="Bộ ba theo frame")
        summary = gr.Markdown()
        btn.click(infer, [vin, fps, thr, topk], [vout, table, fout, summary])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("DSP_CKPT", str(ROOT / "runs" / "A1" / "ckpt" / "epoch4_weights.pth")))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    a = ap.parse_args()
    ckpt = resolve_ckpt(a.ckpt)
    # máy nhà: chỉ 127.0.0.1; Space tự đặt GRADIO_SERVER_NAME (0.0.0.0) và tự kiểm soát quyền truy cập (Space private)
    build(ckpt).queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=a.port, share=False)
