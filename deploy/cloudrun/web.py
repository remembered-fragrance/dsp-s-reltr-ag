"""Web demo cho Google Cloud Run: FastAPI + một trang HTML, đăng nhập HTTP Basic.

Không dùng Gradio ở đây: Gradio giữ một kết nối heartbeat mở suốt lúc tab còn mở, và Cloud Run (tính tiền theo
request) sẽ tính tiền cả khi không ai dùng. Trang này chỉ gửi request khi bấm "Chạy".

  DSP_DEMO_PASS=... uvicorn web:app --port 8080      # trong image (thư mục /app)
Biến môi trường: DSP_DEMO_USER (mặc định dsp), DSP_DEMO_PASS (bắt buộc), DSP_CKPT, DSP_MAX_SECONDS (60), DSP_MAX_MB (30).
"""
import base64
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in (HERE, *HERE.parents) if (p / "tools" / "demo.py").exists())  # /app trong image, gốc repo khi chạy từ repo
sys.path.insert(0, str(ROOT / "tools"))
import demo as D  # noqa: E402

USER = os.environ.get("DSP_DEMO_USER", "dsp")
PASS = (os.environ.get("DSP_DEMO_PASS") or "").rstrip("\r\n")
if len(PASS) < 8:
    raise SystemExit("cần DSP_DEMO_PASS (ít nhất 8 ký tự)")
CKPT = os.environ.get("DSP_CKPT", str(ROOT / "weights" / "epoch4_weights.pth"))
MAX_SECONDS = float(os.environ.get("DSP_MAX_SECONDS", "60"))
MAX_MB = float(os.environ.get("DSP_MAX_MB", "30"))  # Cloud Run giới hạn request HTTP/1 ở 32 MiB
JOBS = Path(tempfile.gettempdir()) / "dsp_jobs"
JOBS.mkdir(exist_ok=True)
LOCK = threading.Lock()  # một video mỗi lần (CPU nhỏ, max-instances 1)
OUTPUTS = {"annotated.mp4": "video/mp4", "graphs.json": "application/json"}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def _authorized(header):
    if not header.lower().startswith("basic "):
        return False
    try:
        user, _, pw = base64.b64decode(header[6:]).decode("utf-8").partition(":")
    except Exception:
        return False
    return secrets.compare_digest(user.encode(), USER.encode()) & secrets.compare_digest(pw.encode(), PASS.encode())


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if request.url.path == "/healthz" or _authorized(request.headers.get("authorization", "")):
        return await call_next(request)
    return Response("Cần đăng nhập", status_code=401, media_type="text/plain; charset=utf-8",
                    headers={"WWW-Authenticate": 'Basic realm="DSP_S demo", charset="UTF-8"'})


@app.get("/healthz")
def healthz():
    return {"ok": True}


def _cleanup(max_age_s=3600):
    for d in JOBS.iterdir():
        if d.is_dir() and time.time() - d.stat().st_mtime > max_age_s:
            shutil.rmtree(d, ignore_errors=True)


@app.post("/api/run")
def run_job(video: UploadFile = File(...), fps: float = Form(1.0), thr: float = Form(0.3), topk: int = Form(10)):
    if not LOCK.acquire(blocking=False):
        raise HTTPException(429, "Máy chủ đang xử lý một video khác, thử lại sau ít phút.")
    try:
        _cleanup()
        fps, thr, topk = min(max(fps, 0.5), 5.0), min(max(thr, 0.05), 0.9), int(min(max(topk, 1), 20))
        jid = uuid.uuid4().hex[:16]
        d = JOBS / jid
        d.mkdir()
        suffix = (Path(video.filename or "").suffix.lower() or ".mp4")[:6]
        src = d / f"input{suffix}"
        size = 0
        with open(src, "wb") as f:
            while chunk := video.file.read(1 << 20):
                size += len(chunk)
                if size > MAX_MB * 2 ** 20:
                    shutil.rmtree(d, ignore_errors=True)
                    raise HTTPException(413, f"Video lớn hơn {MAX_MB:.0f} MB.")
                f.write(chunk)
        t0 = time.time()
        try:
            _, _, graphs = D.run(str(src), CKPT, d, fps=fps, thr=thr, topk=topk, max_seconds=MAX_SECONDS,
                                 source=f"web Cloud Run ({Path(video.filename or 'video').name})")
        except (AssertionError, subprocess.CalledProcessError):
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(400, "Không đọc được video (định dạng không hỗ trợ hoặc file hỏng).")
        finally:
            src.unlink(missing_ok=True)
        rows = [[g["time_s"], t["predicate"], t["object"], t["score"]] for g in graphs for t in g["triplets"]]
        return {"job": jid, "seconds": round(time.time() - t0, 1), "frames": len(graphs), "max_seconds": MAX_SECONDS,
                "video": f"api/jobs/{jid}/annotated.mp4", "graphs": f"api/jobs/{jid}/graphs.json", "rows": rows}
    finally:
        LOCK.release()


@app.get("/api/jobs/{jid}/{name}")
def job_file(jid: str, name: str):
    if not re.fullmatch(r"[0-9a-f]{16}", jid) or name not in OUTPUTS:
        raise HTTPException(404)
    p = JOBS / jid / name
    if not p.exists():
        raise HTTPException(404, "Kết quả đã hết hạn (giữ 1 giờ) hoặc máy chủ đã khởi động lại.")
    return FileResponse(p, media_type=OUTPUTS[name], filename=name)


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("{{MAX_MB}}", f"{MAX_MB:.0f}").replace("{{MAX_SECONDS}}", f"{MAX_SECONDS:.0f}")


PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DSP_S — video thành đồ thị cảnh</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#141414;--ink2:#55534e;--line:#e3e2dc;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--card:#1f1f1d;--ink:#f2f2f0;--ink2:#b9b8b0;--line:#33332f;--accent:#4a90e2}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--ink2);margin:0 0 20px}
.grid{display:grid;grid-template-columns:minmax(0,340px) minmax(0,1fr);gap:16px}@media (max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
label{display:block;font-size:13px;color:var(--ink2);margin:12px 0 4px}input[type=file]{width:100%}
input[type=range]{width:100%}.row{display:flex;justify-content:space-between;font-size:13px}
button{margin-top:16px;width:100%;padding:10px;border:0;border-radius:8px;background:var(--accent);color:#fff;font-weight:600;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}#status{margin-top:12px;font-size:13px;color:var(--ink2);min-height:20px}
video{width:100%;border-radius:8px;background:#000}table{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line)}th{color:var(--ink2);font-weight:600}
.tablewrap{max-height:340px;overflow:auto}.err{color:#d03b3b}a{color:var(--accent)}.empty{color:var(--ink2);font-size:14px}
</style></head><body><main>
<h1>Video → đồ thị cảnh (RelTR baseline A1)</h1>
<p class="sub">Mô hình RelTR fine-tune trên Action Genome. Chạy CPU: khoảng 1–2 giây mỗi frame; chỉ xử lý {{MAX_SECONDS}} giây đầu,
video tối đa {{MAX_MB}} MB. Demo minh họa, không dùng số liệu trong báo cáo.</p>
<div class="grid">
<form class="card" id="f">
<label for="video">Video (mp4, mov, webm…)</label><input id="video" name="video" type="file" accept="video/*" required>
<label>Số frame mỗi giây</label><div class="row"><span>0,5</span><b id="fpsv">1</b><span>5</span></div>
<input name="fps" id="fps" type="range" min="0.5" max="5" step="0.5" value="1">
<label>Ngưỡng điểm bộ ba</label><div class="row"><span>0,05</span><b id="thrv">0,30</b><span>0,9</span></div>
<input name="thr" id="thr" type="range" min="0.05" max="0.9" step="0.05" value="0.3">
<label>Tối đa bộ ba mỗi frame</label><div class="row"><span>1</span><b id="topkv">10</b><span>20</span></div>
<input name="topk" id="topk" type="range" min="1" max="20" step="1" value="10">
<button id="go" type="submit">Chạy</button><div id="status"></div>
</form>
<section class="card" id="out"><p class="empty">Kết quả hiện ở đây: video đã chú thích và bảng bộ ba theo thời gian.</p></section>
</div></main>
<script>
const vn = x => String(x).replace(".", ",");
for (const [id, d] of [["fps",1],["thr",2],["topk",0]]) {
  const el = document.getElementById(id), out = document.getElementById(id + "v");
  const show = () => out.textContent = vn(Number(el.value).toFixed(d)); el.addEventListener("input", show); show();
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
document.getElementById("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = document.getElementById("video").files[0], status = document.getElementById("status"), go = document.getElementById("go");
  if (!file) return;
  if (file.size > {{MAX_MB}} * 1024 * 1024) { status.innerHTML = '<span class="err">Video lớn hơn {{MAX_MB}} MB.</span>'; return; }
  go.disabled = true; const t0 = Date.now();
  const tick = setInterval(() => status.textContent = "Đang xử lý… " + Math.round((Date.now() - t0) / 1000) + " giây", 500);
  try {
    const r = await fetch("api/run", {method: "POST", body: new FormData(e.target)});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || ("Lỗi " + r.status));
    const rows = j.rows.map(r => `<tr><td>${vn(r[0])}</td><td>${esc(r[1])}</td><td>${esc(r[2])}</td><td>${vn(Number(r[3]).toFixed(2))}</td></tr>`).join("");
    document.getElementById("out").innerHTML =
      `<video controls src="${j.video}"></video>
       <p class="sub" style="margin:8px 0 0">${j.frames} frame (tối đa ${j.max_seconds} giây đầu), ${vn(j.seconds)} giây xử lý ·
       <a href="${j.video}" download>tải video</a> · <a href="${j.graphs}" download>tải graphs.json</a></p>
       <div class="tablewrap"><table><thead><tr><th>Thời điểm (s)</th><th>Quan hệ</th><th>Vật</th><th>Điểm</th></tr></thead>
       <tbody>${rows || '<tr><td colspan="4">Không có bộ ba nào vượt ngưỡng.</td></tr>'}</tbody></table></div>`;
    status.textContent = "Xong sau " + Math.round((Date.now() - t0) / 1000) + " giây.";
  } catch (err) { status.innerHTML = '<span class="err">' + esc(err.message) + "</span>"; }
  finally { clearInterval(tick); go.disabled = false; }
});
</script></body></html>"""
