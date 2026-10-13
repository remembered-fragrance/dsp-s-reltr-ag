# DSP_S — baseline RelTR trên Action Genome

Mã nguồn của đợt chạy baseline **A1**: RelTR nguyên bản fine-tune trên Action Genome (train trên Kaggle 2× T4),
đánh giá SGDet theo giao thức STTran, cùng demo *video → đồ thị cảnh* chạy trên Hugging Face Space private.
Kết quả A1 (test AG-full): R@20 With Constraint = 22,6.

Repo **không** chứa dữ liệu (Charades, Action Genome), trọng số, kết quả chạy hay mã RelTR.
RelTR được dựng lại từ upstream (commit `fca7397`) cộng bản vá `patches/reltr_dsp.patch`.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `tools/` | Pipeline: trích frame, chuyển nhãn, dự đoán, chấm điểm (port STTran), báo cáo, lưu/resume, demo |
| `patches/reltr_dsp.patch` | Bản vá P1–P13 cho RelTR (lớp AG, AMP, DDP, resume, hạn giờ, chấm dev) |
| `configs/` | `A1.yaml` (mẫu) và `A1_decided_G5.yaml` (cấu hình đã chốt của A1) |
| `demo/app.py` | Giao diện Gradio (máy nhà hoặc Space) |
| `space/` | Đóng gói và deploy lên Hugging Face Space |
| `tests/` | Kiểm thử CPU |
| `.github/workflows/ci-cd.yml` | CI (kiểm thử) và CD (deploy Space) |

## Cài đặt

```bash
python tools/setup_reltr.py          # clone RelTR fca7397 + áp bản vá -> third_party/RelTR
pip install -r requirements.txt      # CPU; máy có GPU cài torch bản CUDA tương ứng
```

Pipeline đầy đủ (máy nhà + Kaggle) cần dữ liệu đặt trong `data/`; các notebook Kaggle sinh bằng `python tools/make_notebooks.py`.

## CI/CD và demo private

- **CI**: mọi push và pull request chạy `tests/test_resume_cpu.py`, `tests/test_model_cpu.py`, `tests/test_push_state_fake.py`,
  kiểm tra dựng giao diện demo và đóng gói Space.
- **CD**: push lên `main` mà CI qua thì đóng gói demo (`space/build_space.py`) và đồng bộ sang Space **private** (`space/deploy_space.py`).
  Script dừng nếu Space đang public.

Thiết lập một lần:

1. Trên Hugging Face: tạo organization, mời thành viên; tạo repo model private `<org>/dsp-s-reltr-a1` và upload `epoch4_weights.pth`.
2. Tạo hai token fine-grained: một token **ghi** cho Space (dùng ở GitHub), một token **chỉ đọc** repo model (dùng trong Space).
3. GitHub → Settings → Secrets and variables → Actions: secret `HF_TOKEN` (token ghi); variables `HF_SPACE_ID` = `<org>/dsp-s-demo`, `HF_MODEL_REPO` = `<org>/dsp-s-reltr-a1`.
4. Sau lần deploy đầu: Space → Settings → secret `HF_TOKEN` (token chỉ đọc).

Space mặc định chạy CPU (khoảng 1–2 giây cho mỗi frame) và chỉ xử lý 60 giây đầu video (`DSP_MAX_SECONDS`).

## Giấy phép dữ liệu

Charades và Action Genome chỉ dùng cho nghiên cứu phi thương mại; RelTR upstream không kèm file LICENSE.
Vì vậy repo, repo model và Space đều để private.
