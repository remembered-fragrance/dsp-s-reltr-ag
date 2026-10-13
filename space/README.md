---
title: DSP_S RelTR Action Genome Demo
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: demo/app.py
pinned: false
---

# DSP_S — video → đồ thị cảnh (RelTR baseline A1)

Demo **private**: RelTR fine-tune trên Action Genome (cấu hình A1), chạy SGDet theo từng frame của video tải lên.
Space này được GitHub Actions đồng bộ tự động; đừng sửa trực tiếp ở đây.

Cấu hình Space (Settings → Variables and secrets):

| Loại | Tên | Giá trị |
|---|---|---|
| Secret | `HF_TOKEN` | token chỉ đọc, có quyền đọc repo model private |
| Variable | `DSP_MODEL_REPO` | `<org>/dsp-s-reltr-a1` (deploy tự đặt) |
| Variable (tùy chọn) | `DSP_MAX_SECONDS` | số giây đầu video được xử lý, mặc định 60 |

Chỉ để minh họa; không dùng số liệu demo trong báo cáo. Dữ liệu Charades/Action Genome chỉ dùng cho nghiên cứu phi thương mại.
