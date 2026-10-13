# DSP_S — baseline RelTR trên Action Genome

Mã nguồn của đợt chạy baseline **A1**: RelTR nguyên bản fine-tune trên Action Genome (train trên Kaggle 2× T4),
đánh giá SGDet theo giao thức STTran, cùng demo *video → đồ thị cảnh* (máy nhà + Cloudflare Tunnel, có mật khẩu).
Kết quả A1 (test AG-full): R@20 With Constraint = 22,6.

Repo **không** chứa dữ liệu (Charades, Action Genome), trọng số, kết quả chạy hay mã RelTR.
RelTR được dựng lại từ upstream (commit `fca7397`) cộng bản vá `patches/reltr_dsp.patch`.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `tools/` | Pipeline: trích frame, chuyển nhãn, dự đoán, chấm điểm (port STTran), báo cáo, lưu/resume, demo |
| `patches/reltr_dsp.patch` | Bản vá P1–P13 cho RelTR (lớp AG, AMP, DDP, resume, hạn giờ, chấm dev) |
| `configs/` | `A1.yaml` (mẫu) và `A1_decided_G5.yaml` (cấu hình đã chốt của A1) |
| `demo/app.py` | Giao diện Gradio (máy nhà hoặc Space), có đăng nhập bằng mật khẩu |
| `deploy/` | `setup_demo.py` (cài demo trên máy mới), `serve_local.ps1` / `serve_local.sh` (máy nhà + Cloudflare Tunnel), `cloudrun/` (web demo trên Google Cloud Run) |
| `space/` | Đóng gói và deploy lên Hugging Face Space |
| `tests/` | Kiểm thử CPU |
| `.github/workflows/ci-cd.yml` | CI (kiểm thử) và CD (deploy Space) |

## Chạy demo trên máy mới

Cần: Python 3.10–3.13, git, [GitHub CLI](https://cli.github.com) đã `gh auth login` (repo private).
Không cần dữ liệu: trọng số A1 (245 MB) nằm ở Release [`weights-a1`](https://github.com/remembered-fragrance/dsp-s-reltr-ag/releases/tag/weights-a1).

```bash
git clone https://github.com/remembered-fragrance/dsp-s-reltr-ag.git
cd dsp-s-reltr-ag
python deploy/setup_demo.py          # .venv + torch (GPU CUDA 12.8 nếu có nvidia-smi, không thì CPU) + RelTR + trọng số (kiểm sha256)
```

Sau đó:

| Mục đích | Windows | Linux / macOS |
|---|---|---|
| Chỉ dùng trên máy đó | `.venv\Scripts\python.exe demo/app.py` | `.venv/bin/python demo/app.py` |
| Mở link cho máy khác (có mật khẩu) | `powershell -ExecutionPolicy Bypass -File deploy/serve_local.ps1` | `bash deploy/serve_local.sh` |

Demo mở ở http://127.0.0.1:7860. GPU: vài giây cho mỗi video; CPU: khoảng 1–2 giây cho mỗi frame.

## Cài đặt cho pipeline đầy đủ

```bash
python tools/setup_reltr.py          # clone RelTR fca7397 + áp bản vá -> third_party/RelTR
pip install -r requirements.txt      # CPU; máy có GPU cài torch bản CUDA tương ứng
```

Pipeline đầy đủ (máy nhà + Kaggle) cần dữ liệu đặt trong `data/`; các notebook Kaggle sinh bằng `python tools/make_notebooks.py`.

## CI và demo

- **CI**: mọi push và pull request chạy `tests/test_resume_cpu.py`, `tests/test_model_cpu.py`, `tests/test_push_state_fake.py`,
  kiểm tra dựng giao diện demo và đóng gói Space.
- **Demo cho máy khác truy cập (miễn phí)**: chạy trên máy nhà (GPU) qua Cloudflare Quick Tunnel, có mật khẩu.
  Script tự kéo code mới nhất của `main` trước khi khởi động.

```powershell
winget install --id Cloudflare.cloudflared                              # một lần
powershell -ExecutionPolicy Bypass -File deploy/serve_local.ps1         # hỏi mật khẩu, in link https://….trycloudflare.com
```

  Gửi link + user `dsp` + mật khẩu cho người cần xem. Link đổi mỗi lần chạy; máy nhà phải bật. Không có cloudflared thì
  dùng `-Share` (link `*.gradio.live` tạm thời). Demo chỉ xử lý 60 giây đầu video (`DSP_MAX_SECONDS`).
- **Web demo trên Google Cloud Run (link cố định, miễn phí trong hạn mức tháng)**: FastAPI + trang HTML
  (`deploy/cloudrun/web.py`, không dùng Gradio vì kết nối heartbeat của Gradio làm Cloud Run tính tiền cả khi không ai dùng),
  đăng nhập HTTP Basic (user `dsp`, mật khẩu ở Secret Manager). CPU 2 vCPU / 4 GiB, scale về 0, tối đa 1 instance,
  mỗi lần một video (≤ 30 MB, 60 giây đầu). Push lên `main` thì job `deploy-cloudrun` build image (kèm trọng số từ Release)
  và deploy; link hiện ở trang tóm tắt của lần chạy GitHub Actions. Thiết lập một lần:

  1. Tạo project ở https://console.cloud.google.com và liên kết tài khoản thanh toán (bắt buộc kể cả khi dùng miễn phí);
     nên đặt cảnh báo ngân sách (Billing → Budgets).
  2. Cài gcloud (`winget install Google.CloudSDK`), mở terminal mới, `gcloud auth login`.
  3. `powershell -ExecutionPolicy Bypass -File deploy/cloudrun/setup_gcp.ps1 -ProjectId <id-project>`
     (hỏi mật khẩu web, tự ghi biến `GCP_*` vào GitHub).
  4. `gh workflow run ci-cd.yml -R remembered-fragrance/dsp-s-reltr-ag` hoặc push lên `main`.

- **Hugging Face Space (tùy chọn, trả phí)**: Space Gradio cần gói PRO (cá nhân) hoặc Team (organization).
  Khi có, đặt biến `HF_SPACE_ID`, `HF_MODEL_REPO` và secret `HF_TOKEN` trên GitHub thì job `deploy` tự đồng bộ
  (`space/build_space.py`, `space/deploy_space.py`); chưa đặt `HF_SPACE_ID` thì job được bỏ qua.
  Trọng số đã có ở repo model private `DSPFA26MiMi/dsp-s-reltr-a1`.

## Giấy phép dữ liệu

Charades và Action Genome chỉ dùng cho nghiên cứu phi thương mại; RelTR upstream không kèm file LICENSE.
Vì vậy repo và repo model để private, demo có mật khẩu.
