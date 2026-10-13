# DSP_S — baseline RelTR trên Action Genome

Demo *video → đồ thị cảnh theo từng frame* bằng RelTR fine-tune trên Action Genome (baseline A1;
test AG-full: R@20 With Constraint = 22,6).

## Cài đặt

Cần:

- Python 3.10–3.13, git, [GitHub CLI](https://cli.github.com) đã `gh auth login`.
- Khoảng 8 GB trống (`.venv` có PyTorch bản CUDA ~5 GB, bộ nhớ đệm pip lúc cài, trọng số A1 245 MB; bản CPU nhỏ hơn nhiều).
  Không cần dữ liệu Charades/Action Genome.

```bash
gh repo clone remembered-fragrance/dsp-s-reltr-ag
cd dsp-s-reltr-ag
python deploy/setup_demo.py          # .venv + torch (GPU CUDA 12.8 nếu có nvidia-smi, không thì CPU) + RelTR + trọng số (kiểm sha256)
```

Thêm `--cpu` để ép bản CPU. Windows không có lệnh `python` thì dùng `py deploy/setup_demo.py`.

## Chạy demo

```bash
.venv/bin/python demo/app.py           # Linux / macOS
.venv\Scripts\python.exe demo/app.py   # Windows
```

Khi terminal in địa chỉ http://127.0.0.1:7860 (sau khi nạp model), mở địa chỉ đó trong trình duyệt.

1. Kéo một video (mp4) vào ô *Video đầu vào*. Model học từ Action Genome nên hợp nhất với video quay **trong nhà, có người
   tương tác với đồ vật** (ngồi ghế, cầm cốc, mở tủ…). Chỉ 60 giây đầu được xử lý.
2. Chỉnh nếu cần: số frame mỗi giây, ngưỡng điểm bộ ba, số bộ ba tối đa mỗi frame. Bấm **Chạy**.
3. Kết quả: video có khung và nhãn *person — quan hệ — vật*, bảng bộ ba theo thời điểm, file `graphs.json`.
   Bản sao (cùng `summary.md`) lưu ở `demo_out/app_<thời gian>/`.

Tốc độ: GPU RTX 5060 khoảng 12 giây cho video 30 giây ở 2 frame/giây; CPU khoảng 1–2 giây cho mỗi frame
(mặc định 1 frame/giây trên CPU).

Không cần giao diện thì chạy dòng lệnh (xử lý cả video; Windows thay `.venv/bin/python` bằng `.venv\Scripts\python.exe`):

```bash
.venv/bin/python tools/demo.py --video <video.mp4> --ckpt runs/A1/ckpt/epoch4_weights.pth --out demo_out/thu --fps 2
```

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `gh repo clone` hoặc tải trọng số báo 404 / không tìm thấy | Tài khoản chưa có quyền đọc repo: nhờ chủ repo mời, chấp nhận lời mời, chạy lại |
| Windows: `OSError: [WinError 4551] An Application Control policy has blocked this file` khi `import torch` | Smart App Control của Windows 11 chặn DLL của PyTorch. Chủ máy tự quyết định tắt ở Windows Security → App & browser control → Smart App Control |
| Lỗi CUDA khi bấm Chạy (vd `no kernel image is available` trên GPU đời cũ) | Chạy bằng CPU: đặt biến `CUDA_VISIBLE_DEVICES=-1` trước khi chạy demo, hoặc xóa `.venv` rồi `python deploy/setup_demo.py --cpu` |
| Cổng 7860 đang bận | `demo/app.py --port 7870` |

Charades và Action Genome chỉ dùng cho nghiên cứu phi thương mại; không dùng demo và trọng số cho mục đích thương mại.
