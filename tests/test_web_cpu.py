"""Kiểm web demo Cloud Run (deploy/cloudrun/web.py): đăng nhập, trang chủ, chặn đường dẫn lạ, giới hạn dung lượng.
Không chạy model (CI không có trọng số). Chạy: python tests/test_web_cpu.py"""
import base64
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["DSP_DEMO_PASS"] = "mat-khau-ci-123"
os.environ.setdefault("DSP_CKPT", "khong-co-trong-so")
sys.path.insert(0, str(ROOT / "deploy" / "cloudrun"))

from fastapi.testclient import TestClient  # noqa: E402

import web  # noqa: E402


def basic(user, pw):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


def main():
    c = TestClient(web.app)
    ok, bad = basic("dsp", "mat-khau-ci-123"), basic("dsp", "sai-mat-khau")
    home = c.get("/")
    page = c.get("/", headers=ok)
    web.MAX_MB = 0.001  # ~1 KB
    big = c.post("/api/run", headers=ok, files={"video": ("a.mp4", b"x" * 4096, "video/mp4")})
    checks = {
        "healthz không cần đăng nhập": c.get("/healthz").status_code == 200,
        "trang chủ đòi đăng nhập (401 + WWW-Authenticate)": home.status_code == 401 and "www-authenticate" in home.headers,
        "sai mật khẩu bị chặn": c.get("/", headers=bad).status_code == 401,
        "đúng mật khẩu vào được trang": page.status_code == 200 and "đồ thị cảnh" in page.text,
        "API cũng đòi đăng nhập": c.post("/api/run").status_code == 401,
        "mã job lạ -> 404": c.get("/api/jobs/zzz/annotated.mp4", headers=ok).status_code == 404,
        "tên file không cho phép -> 404": c.get(f"/api/jobs/{'a' * 16}/web.py", headers=ok).status_code == 404,
        "video quá lớn -> 413": big.status_code == 413,
    }
    for k, v in checks.items():
        print(("ok  " if v else "SAI ") + k)
    print("web demo:", "QUA" if all(checks.values()) else "KHÔNG QUA")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
