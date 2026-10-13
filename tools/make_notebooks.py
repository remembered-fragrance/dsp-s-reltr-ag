"""Sinh 4 notebook Kaggle (FR-9) vào notebooks/: 00-prep, 01-checks, 02-train, 03-eval (dự phòng).

  python tools/make_notebooks.py
Sửa nội dung cell ở file này rồi chạy lại; không sửa tay file .ipynb.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"

CELL_T0 = r'''# Cell 1 — T0 và thông tin máy (FR-9, SR-1, SR-2)
import json, os, shutil, subprocess, sys, time
from pathlib import Path
T0 = time.time()
DEADLINE_TRAIN = int(T0 + 10*3600 + 45*60)   # SR-2
SESSION_END = T0 + 11*3600 + 30*60           # SR-1
WORK = Path('/kaggle/working')
TMP = Path('/kaggle/tmp')
try:
    TMP.mkdir(parents=True, exist_ok=True)
except OSError:
    TMP = Path('/tmp/dsp'); TMP.mkdir(parents=True, exist_ok=True)

def sh(cmd, check=True, env=None):
    print('$', cmd, flush=True)
    rc = subprocess.run(cmd, shell=True, env=env).returncode
    if check and rc != 0:
        raise RuntimeError(f'lỗi (mã {rc}): {cmd}')
    return rc

def find_dir(root, marker, max_depth=4):
    """Thư mục chứa file `marker` (tìm nông trước, tránh duyệt hết thư mục frame)."""
    for d in range(max_depth + 1):
        hits = sorted(Path(root).glob('*/' * d + marker))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(f'không thấy {marker} trong {root} — kiểm tra đã gắn input chưa')

# Kaggle gắn dataset ở /kaggle/input/datasets/<user>/<slug> (bố cục mới); dự phòng /kaggle/input/<slug> (bố cục cũ)
KAGGLE_USER = 'nguyntaif'
INPUT = Path('/kaggle/input')
BIG_INPUTS = {'dsp-code', 'dsp-ag-annotations', 'reltr-vg-ckpt', 'dsp-ag-data'}

def ds(slug, required=True):
    for p in (INPUT / 'datasets' / KAGGLE_USER / slug, INPUT / slug):
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(f'chưa gắn dataset {slug}: Add Input → {KAGGLE_USER}/{slug}')
    return None

def mounted():
    """Mọi thư mục đã gắn (dataset, output notebook), cả bố cục mới lẫn cũ."""
    out = []
    for p in sorted(INPUT.iterdir()) if INPUT.exists() else []:
        if p.name in ('datasets', 'notebooks', 'models', 'competitions') and p.is_dir():
            out += [q for u in sorted(p.iterdir()) if u.is_dir() for q in sorted(u.iterdir()) if q.is_dir()]
        elif p.is_dir():
            out.append(p)
    return out

info = subprocess.run(f"nvidia-smi; echo; nproc; free -g; df -h {WORK} {TMP}; find {INPUT} -maxdepth 3 -type d; "
                      "python -c 'import torch;print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"gpus\", torch.cuda.device_count())'",
                      shell=True, capture_output=True, text=True).stdout
print(info)
fmt = lambda t: time.strftime('%Y-%m-%d %H:%M', time.localtime(t))
print(f'T0 {fmt(T0)} | hạn train {fmt(DEADLINE_TRAIN)} | hạn phiên {fmt(SESSION_END)} | TMP {TMP}')'''

CELL_CODE = r'''# Cell 2 — chuẩn bị code: chép dsp-code sang vùng tạm (input chỉ đọc)
CODE = TMP / 'code'
shutil.rmtree(CODE, ignore_errors=True)
shutil.copytree(ds('dsp-code'), CODE)   # /kaggle/input/datasets/nguyntaif/dsp-code
CODE = find_dir(CODE, 'CODE_VERSION.json', 2)   # phòng khi Kaggle lồng thêm một cấp thư mục
for z in list(CODE.glob('*.zip')):   # dataset tải lên bằng --dir-mode zip mà Kaggle chưa tự giải nén
    shutil.unpack_archive(str(z), str(CODE / z.stem)); z.unlink()
CODE_VERSION = json.loads((CODE / 'CODE_VERSION.json').read_text())
print('dsp-code:', CODE_VERSION)
sh('python -c "import pycocotools" || pip -q install pycocotools', check=False)
RELTR = CODE / 'third_party' / 'RelTR'
TOOLS = CODE / 'tools'
CHECKS = WORK / 'checks'; CHECKS.mkdir(parents=True, exist_ok=True)'''

CELL_SECRETS = r'''# Cell — token Kaggle cho đẩy trạng thái (require 5.6, 9.6). Không in, không ghi ra file.
PUSH = False
try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    def _get(name):
        try:
            return _s.get_secret(name)
        except Exception:
            return None
    _user, _tok, _key = _get('KAGGLE_USERNAME'), _get('KAGGLE_API_TOKEN'), _get('KAGGLE_KEY')
    if _user and (_tok or _key):
        os.environ['KAGGLE_USERNAME'] = _user
        if _tok:
            os.environ['KAGGLE_API_TOKEN'] = _tok
        if _key:
            os.environ['KAGGLE_KEY'] = _key
        PUSH = True
except Exception as e:
    print('Không đọc được Kaggle Secrets:', type(e).__name__)
if PUSH:  # token kiểu mới (KGAT_...) cần kaggle CLI >= 2.0; thử gọi API (không in token)
    subprocess.run('pip -q install -U "kaggle>=2.0"', shell=True)
    _r = subprocess.run(['kaggle', 'datasets', 'list', '--mine'], capture_output=True, text=True)
    # CLI 2.x có thể trả mã 0 cả khi token sai: chỉ coi là OK khi thấy dataset của chính mình trong danh sách
    PUSH = _r.returncode == 0 and f"{_user}/dsp-code" in _r.stdout
    print('Kiểm tra token với Kaggle API:', 'OK' if PUSH else 'LỖI — ' + (_r.stdout + _r.stderr).strip()[-300:])
print('Đẩy trạng thái:', 'BẬT (đã nạp token từ Secrets)' if PUSH else 'TẮT — thiếu hoặc sai KAGGLE_USERNAME + KAGGLE_API_TOKEN/KAGGLE_KEY; dựa vào output của commit')'''

CELL_INPUTS = r'''# Cell — tìm dữ liệu đã gắn (dsp-ag-data, dsp-ag-annotations, reltr-vg-ckpt)
ANN = find_dir(ds('dsp-ag-annotations'), 'person_bbox.pkl', 2)        # /kaggle/input/datasets/nguyntaif/dsp-ag-annotations
CKPT_VG = find_dir(ds('reltr-vg-ckpt'), 'checkpoint0149.pth', 2) / 'checkpoint0149.pth'
DATA = ds('dsp-ag-data')
FRAMES = find_dir(DATA, '_keep_list.txt', 3)
DATA_RELTR = find_dir(DATA, 'dev300.txt', 3)
OVERFIT = DATA_RELTR.parent / (DATA_RELTR.name + '_overfit')
print(f'ANN {ANN}\nCKPT_VG {CKPT_VG}\nFRAMES {FRAMES}\nDATA_RELTR {DATA_RELTR}\nOVERFIT {OVERFIT}')'''


def md(s):
    return ("markdown", s)


def code(s):
    return ("code", s)


NB00 = [
    md("# 00-prep — tải video, trích frame, chuyển dữ liệu, G1–G2 (chỉ CPU)\n\n"
       "Chạy bằng **Save Version → Save & Run All**, accelerator **None** (không tốn quota GPU). Bật Internet.\n"
       "Input: `dsp-code`, `dsp-ag-annotations`. Xong: tạo dataset **`dsp-ag-data`** từ output của phiên bản này "
       "(Output → New Dataset), để private."),
    code(CELL_T0), code(CELL_CODE),
    code(r'''# Kiểm dung lượng vùng tạm (require 6.1): cần >= 20GB
ANN = find_dir(ds('dsp-ag-annotations'), 'person_bbox.pkl', 2)   # /kaggle/input/datasets/nguyntaif/dsp-ag-annotations
print('ANN', ANN)
free_tmp = shutil.disk_usage(TMP).free / 1e9; free_work = shutil.disk_usage(WORK).free / 1e9
print(f'trống: TMP {free_tmp:.1f} GB, working {free_work:.1f} GB')
assert free_tmp >= 20, 'Vùng tạm < 20GB: trích frame ở máy nhà rồi upload thư mục frame làm dataset (require 6.1)'
assert free_work >= 8, 'working còn < 8GB: không đủ chỗ cho frame'
'''),
    code(r'''# Tải Charades 480p (16,3GB) vào vùng tạm — không lưu vào output
URL = 'https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip'
ZIP = TMP / 'Charades_v1_480.zip'
t = time.time()
sh(f'wget -c -q -O {ZIP} {URL}')
assert ZIP.stat().st_size == 16339546533, f'kích thước zip sai: {ZIP.stat().st_size}'
print(f'tải xong trong {(time.time() - t) / 60:.1f} phút')'''),
    code(r'''# FR-1: trích frame (test + dev + k_max=8 frame/video train), chạy song song, có hạn giờ (SR-5)
FRAMES = WORK / 'data' / 'frames'
sh(f'python {TOOLS}/extract_frames.py --video_zip {ZIP} --ann_dir {ANN} --out_dir {FRAMES} '
   f'--splits train dev test --workers {os.cpu_count()} --tmp_dir {TMP} --deadline_unix {int(T0 + 11*3600)}')
print(json.loads((FRAMES / '_extract_status.json').read_text()))'''),
    code(r'''# G1: chỉ số frame
sh(f'python {TOOLS}/g1_check.py --video_zip {ZIP} --frames_dir {FRAMES} --n 3 --out {CHECKS}/g1_frames.csv')'''),
    code(r'''# FR-2 + FR-3 / G2: chuyển dữ liệu (k=8 tạm thời; k thật chốt ở G5) + overfit 64 + thống kê và hình
RELTR_DATA = WORK / 'data' / 'reltr'
sh(f'python {TOOLS}/convert_ag.py --ann_dir {ANN} --frames_dir {FRAMES} --out_dir {RELTR_DATA} --k 8 --overfit 64')
sh(f'python {TOOLS}/check_data.py --ann_path {RELTR_DATA} --img_folder {FRAMES} --split test --out_dir {CHECKS}/data_viz --stats {CHECKS}/g2_stats.txt')
sh(f'python {TOOLS}/check_data.py --ann_path {RELTR_DATA} --img_folder {FRAMES} --split train --n 10 --out_dir {CHECKS}/data_viz_train --stats {CHECKS}/g2_stats_train.txt')
print((RELTR_DATA / 'meta.json').read_text()[:2000])'''),
    code(r'''# G2: TỰ MỞ XEM vài ảnh — box có đúng vị trí, nhãn có đúng vật không
from IPython.display import Image as IPImage, display
for p in sorted((CHECKS / 'data_viz').glob('*.jpg'))[:6]:
    print(p.name); display(IPImage(filename=str(p), width=480))'''),
    code(r'''# Kết thúc: dọn vùng tạm, tóm tắt
ZIP.unlink(missing_ok=True)
sh(f'du -sh {WORK}/data/* {CHECKS}', check=False)
print(f'Thời gian phiên: {(time.time() - T0) / 3600:.2f} giờ')
print('Bước tiếp: Output → New Dataset → tên dsp-ag-data (private). Sau đó chạy 01-checks.')
assert time.time() < SESSION_END, 'Vượt SR-1'
'''),
]

NB01 = [
    md("# 01-checks — G0, G5, G6 (GPU T4 x2) + kiểm bằng chứng G4 (chạy ở máy nhà)\n\n"
       "Chạy bằng **Save Version → Save & Run All** (tự chạy hết, khoảng 1 giờ GPU), accelerator **GPU T4 x2**, bật Internet. "
       "Input: `dsp-code`, `dsp-ag-data`, `dsp-ag-annotations`, `reltr-vg-ckpt`. Secrets (tùy chọn): `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`.\n\n"
       "Mỗi cổng phải **QUA** mới sang cổng sau (MR-10): cổng không qua thì notebook dừng ở đó.\n\n"
       "Output của notebook này chứa `configs/A1.yaml` đã chốt ở G5: **gắn output này làm input của `02-train`** "
       "(không cần tải về hay upload lại `dsp-code`)."),
    code(CELL_T0), code(CELL_CODE), code(CELL_SECRETS), code(CELL_INPUTS),
    code(r'''# G0: 2× T4, dung lượng, checkpoint VG cho bộ ba hợp lý
import torch
n = torch.cuda.device_count(); names = [torch.cuda.get_device_name(i) for i in range(n)]
free_work = shutil.disk_usage(WORK).free / 1e9
sh(f'python {TOOLS}/g0_vg_inference.py --ckpt {CKPT_VG} --out {CHECKS}/g0_vg.txt')
(CHECKS / 'g0_env.txt').write_text(info + f'\nGPU: {names}\nworking trống: {free_work:.1f} GB\n')
ok = n == 2 and all('T4' in x for x in names) and free_work >= 15
print('G0', 'QUA' if ok else 'KHÔNG QUA', names, f'{free_work:.1f} GB'); assert ok'''),
    code(r'''# G4 chạy ở máy nhà (require mục 10, nhật ký 2026-10-12): kiểm bằng chứng đi kèm dsp-code, chép vào checks/ của notebook này
g4_src = CODE / 'checks' / 'g4_overfit_local.json'
assert g4_src.exists(), 'dsp-code thiếu checks/g4_overfit_local.json (G4 chạy ở máy nhà)'
g4 = json.loads(g4_src.read_text())
sys.path.insert(0, str(TOOLS)); import g4_summary
same_code = g4['code_sha256'] == g4_summary.code_fingerprint(RELTR)
same_data = g4['overfit_data_sha256'] == g4_summary.data_fingerprint(OVERFIT)
shutil.copyfile(g4_src, CHECKS / 'g4_overfit.json')
print('G4 (máy nhà):', 'QUA' if g4['pass'] else 'KHÔNG QUA', g4['trajectory_R@20_no'], '| code khớp:', same_code, '| dữ liệu khớp:', same_data)
assert g4['pass'] and same_code and same_data, 'G4 chưa qua, hoặc code train / dữ liệu đã đổi sau khi chạy G4 — chạy lại G4 ở máy nhà'
'''),
    code(r'''# G5: đo tốc độ trên 2 GPU (require mục 8). Hàm chạy 220 vòng lặp, ghi GPU util mỗi giây.
N_VIDEOS = json.loads((DATA_RELTR / 'meta.json').read_text())['splits']['train']['videos']
def g5_run(label, lr_backbone, workers=2):
    out = WORK / 'runs' / f'g5_{label}'
    out.parent.mkdir(parents=True, exist_ok=True)
    util = open(out.parent / f'g5_{label}_util.csv', 'w')
    mon = subprocess.Popen('nvidia-smi --query-gpu=timestamp,index,utilization.gpu --format=csv,noheader,nounits -lms 1000',
                           shell=True, stdout=util)
    try:
        sh(f'cd {RELTR} && torchrun --nproc_per_node=2 main.py --dataset ag --ann_path {DATA_RELTR} --img_folder {FRAMES} '
           f'--output_dir {out} --pretrained {CKPT_VG} --lr_backbone {lr_backbone} --batch_size 2 --epochs 4 --lr_drop 3 '
           f'--max_iters 220 --print_freq 1 --amp --no_dev_eval --num_workers {workers} --ckpt_every_min 0')
    finally:
        mon.terminate(); util.close()
    sh(f'python {TOOLS}/g5_speed.py measure --run {out} --label {label} --world 2 --batch 2 '
       f'--gpu_util {out.parent}/g5_{label}_util.csv --out {CHECKS}/g5_{label}.json')
    return json.loads((CHECKS / f'g5_{label}.json').read_text())

op = g5_run('open', 1e-5)
k_open = int(36000 // (op['tau_s_per_frame'] * 4 * N_VIDEOS))
print(f"tau_open={op['tau_s_per_frame']} s/frame, util {op['gpu_util_mean']}%, k_open={k_open}, N={N_VIDEOS}")'''),
    code(r'''# G5 (tiếp): đo đóng băng backbone nếu k_open < 6; đo num_workers 3 nếu GPU util < 70% (quy tắc 2, 4)
fr = g5_run('frozen', 0) if k_open < 6 else None
chosen = op if fr is None else fr
ex = g5_run('w3', 1e-5 if fr is None else 0, workers=3) if (chosen['gpu_util_mean'] or 100) < 70 else None
CFG = WORK / 'configs' / 'A1.yaml'; CFG.parent.mkdir(exist_ok=True); shutil.copyfile(CODE / 'configs' / 'A1.yaml', CFG)
sh(f'python {TOOLS}/g5_speed.py decide --open {CHECKS}/g5_open.json ' + (f'--frozen {CHECKS}/g5_frozen.json ' if fr else '')
   + (f'--extra {CHECKS}/g5_w3.json ' if ex else '') + f'--n_videos {N_VIDEOS} --config {CFG} --out {CHECKS}/g5_speed.txt')
import yaml; cfg = yaml.safe_load(open(CFG)); K = cfg['k']
RELTR_K = TMP / 'data' / f'reltr_k{K}'
sh(f'python {TOOLS}/convert_ag.py --ann_dir {ANN} --frames_dir {FRAMES} --out_dir {RELTR_K} --k {K} --no_size_check')
sh(f'python {TOOLS}/g5_speed.py sha --config {CFG} --train_json {RELTR_K}/train.json')
print(open(CFG).read()); print('G5', 'QUA' if K >= 3 else 'KHÔNG QUA')'''),
    code(r'''# G6: lưu và resume (dữ liệu train thật, 2 GPU, cấu hình A1.yaml vừa chốt)
R = WORK / 'runs'
base = (f'cd {RELTR} && torchrun --nproc_per_node=2 main.py --dataset ag --ann_path {RELTR_K} --img_folder {FRAMES} '
        f'--config {CFG} --amp --seed 42 --no_dev_eval --ckpt_every_min 0 --print_freq 20')
# tham chiếu: chạy liền 60 vòng lặp
sh(f'{base} --output_dir {R}/g6_ref --pretrained {CKPT_VG} --max_iters 60 --log_sample_ids')
# chạy ngắt: 40 vòng lặp (dừng như khi hết giờ) rồi resume tới 60
sh(f'{base} --output_dir {R}/g6_res --pretrained {CKPT_VG} --max_iters 40 --log_sample_ids')
sh(f'{base} --output_dir {R}/g6_res --resume {R}/g6_res/ckpt/last.pth --max_iters 60 --log_sample_ids')
# (a) hết giờ: hạn = bây giờ + 4 phút -> lưu, STATUS time_limit, mã 0
rc_a = sh(f'{base} --output_dir {R}/g6_deadline --pretrained {CKPT_VG} --deadline_unix {int(time.time()) + 240}', check=False)
# (c) cắt cụt last.pth -> restore_state phải chọn last_prev.pth
C6 = TMP / 'g6_corrupt' / 'runs' / 'A1'; shutil.rmtree(C6.parents[1], ignore_errors=True)
shutil.copytree(R / 'g6_res', C6)
raw = (C6 / 'ckpt' / 'last.pth').read_bytes(); (C6 / 'ckpt' / 'last.pth').write_bytes(raw[: len(raw) // 2]); del raw
r = subprocess.run(f'python {TOOLS}/restore_state.py --search {C6.parents[1]} --dst {TMP}/g6_corrupt_dst --tmp {TMP}/g6_unz',
                   shell=True, capture_output=True, text=True); print(r.stdout[-1500:], r.stderr[-1500:])
(CHECKS / 'g6_corrupt_restore.txt').write_text(r.stdout)
# (d) resume từ thư mục chỉ đọc (mô phỏng /kaggle/input)
RO = TMP / 'g6_ro'; shutil.rmtree(RO, ignore_errors=True); shutil.copytree(R / 'g6_res' / 'ckpt', RO); sh(f'chmod -R a-w {RO}')
rc_d = sh(f'{base} --output_dir {R}/g6_ro --resume {RO}/last.pth --max_iters 70', check=False)
# (e) một lần đẩy lên dataset thử (KHÔNG phải dsp-a1-state, để phiên train thật không resume nhầm)
push = 'unused'
if PUSH:
    push = 'ok' if sh(f'python {TOOLS}/push_state.py --run {R}/g6_res --push_dir {TMP}/state_push_test --msg "G6 test" --dataset dsp-a1-state-test', check=False) == 0 else 'fail'
sh(f'python {TOOLS}/g6_check.py --ref {R}/g6_ref --res {R}/g6_res --resume_step 40 --final_step 60 --deadline_run {R}/g6_deadline '
   f'--deadline_rc {rc_a} --corrupt_json {CHECKS}/g6_corrupt_restore.txt --readonly_rc {rc_d} --push {push} --out {CHECKS}/g6_resume.json')'''),
    code(r'''# Kết thúc: dọn checkpoint thử (giữ log và bằng chứng), in A1.yaml để tải về
for d in (WORK / 'runs').glob('*/ckpt'):
    shutil.rmtree(d, ignore_errors=True)
print(open(CFG).read())
gates = {g: json.loads((CHECKS / f).read_text()).get('pass') for g, f in (('G4', 'g4_overfit.json'), ('G6', 'g6_resume.json'))}
print('Cổng:', gates)
print(f'Thời gian phiên: {(time.time() - T0) / 3600:.2f} giờ. Bước tiếp: gắn output notebook này làm input của 02-train.')'''),
]

NB02 = [
    md("# 02-train — train A1 (GPU T4 x2, commit)\n\n"
       "Chạy bằng **Save Version → Save & Run All**, accelerator **GPU T4 x2**, bật Internet.\n"
       "Input: `dsp-code`, `dsp-ag-data`, `dsp-ag-annotations`, `reltr-vg-ckpt`, và **output của `01-checks`** (chứa `configs/A1.yaml` đã chốt); "
       "phiên resume: thêm `dsp-a1-state` (mới nhất) hoặc output của phiên `02-train` trước.\n"
       "Secrets (khuyến nghị): `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`."),
    code(CELL_T0), code(CELL_CODE), code(CELL_SECRETS), code(CELL_INPUTS),
    code(r'''# Cấu hình A1 và sinh lại JSON với k đã chốt (require 6.6): sha256 phải khớp A1.yaml
import yaml
def _decided(p):
    try:
        c = yaml.safe_load(open(p))
        return bool(c and c.get('k') and c.get('train_json_sha256') and c.get('lr_backbone') is not None)
    except Exception:
        return False
# ưu tiên A1.yaml trong output của 01-checks đã gắn; sau đó tới bản trong dsp-code
cands = [q for p in mounted() if p.name not in BIG_INPUTS for q in sorted(p.glob('configs/A1.yaml'))] + [CODE / 'configs' / 'A1.yaml']
decided = [p for p in cands if _decided(p)]
assert decided, f'Không có A1.yaml đã chốt ở G5 (đã gắn output của 01-checks chưa?). Đã xem: {[str(p) for p in cands]}'
assert len({open(p).read() for p in decided}) == 1, f'Có nhiều A1.yaml khác nhau: {[str(p) for p in decided]} — chỉ gắn một output 01-checks'
CFG = decided[0]
cfg = yaml.safe_load(open(CFG))
print('A1.yaml lấy từ', CFG)
# MR-10: chỉ train khi G4, G5, G6 của 01-checks đều qua
G_DIR = CFG.parent.parent / 'checks'
if G_DIR.exists():
    g4 = json.loads((G_DIR / 'g4_overfit.json').read_text())['pass']
    g6 = json.loads((G_DIR / 'g6_resume.json').read_text())['pass']
    g5 = (G_DIR / 'g5_speed.txt').exists()
    print(f'Cổng từ 01-checks: G4 {g4}, G5 {g5}, G6 {g6}')
    assert g4 and g5 and g6, 'Có cổng chưa qua ở 01-checks: không train (MR-10)'
else:
    print('CẢNH BÁO: A1.yaml không đi kèm thư mục checks/ của 01-checks — kiểm tra tay G4–G6 trước khi để chạy tiếp')
RELTR_K = TMP / 'data' / 'reltr'
sh(f'python {TOOLS}/convert_ag.py --ann_dir {ANN} --frames_dir {FRAMES} --out_dir {RELTR_K} --k {cfg["k"]} --no_size_check')
sys.path.insert(0, str(TOOLS)); import ag_common
got = ag_common.sha256_file(RELTR_K / 'train.json')
assert got == cfg['train_json_sha256'], f'train.json sha256 {got} khác A1.yaml — dừng'
print('A1.yaml:', cfg)'''),
    code(r'''# Khôi phục trạng thái (require 9.5)
RUN = WORK / 'runs' / 'A1'
# tìm trong dsp-a1-state và output notebook đã gắn; bỏ qua các dataset lớn không chứa checkpoint
SEARCH = [str(p) for p in mounted() if p.name not in BIG_INPUTS]
print('Tìm checkpoint trong:', SEARCH)
r = subprocess.run(['python', str(TOOLS / 'restore_state.py'), '--search', *SEARCH, '--dst', str(RUN), '--tmp', str(TMP / 'restore'),
                    '--run_name', 'A1'],   # chỉ nhận runs/A1, không bao giờ resume từ run thử G4–G6
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:])
state = json.loads(r.stdout.strip().splitlines()[-1])
print('trạng thái:', state)'''),
    code(r'''# Train và kết thúc trong CÙNG một cell: phần kết thúc luôn chạy kể cả khi train lỗi
cmd = ['torchrun', '--nproc_per_node=2', 'main.py', '--dataset', 'ag', '--ann_path', str(RELTR_K), '--img_folder', str(FRAMES),
       '--output_dir', str(RUN), '--config', str(CFG), '--amp', '--seed', '42', '--deadline_unix', str(DEADLINE_TRAIN),
       '--ckpt_every_min', '30', '--ag_ann_dir', str(ANN), '--print_freq', '100', '--push_dir', str(TMP / 'state_push')]
cmd += ['--push_state'] if PUSH else []
cmd += ['--resume', state['ckpt']] if state['mode'] == 'resume' else ['--pretrained', str(CKPT_VG)]
rc = None
try:
    if state['mode'] != 'done':
        print(' '.join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=str(RELTR)).returncode
finally:
    subprocess.run(['python', str(TOOLS / 'finalize_session.py'), '--run', str(RUN), '--t0', str(T0), '--rc', str(rc), '--wait_push_min', '15'])
    print('Thời gian phiên (giờ):', round((time.time() - T0) / 3600, 2))
    assert time.time() < SESSION_END, 'Vượt SR-1: xem lại deadline'
'''),
    code(r'''# Xem nhanh
print((RUN / 'STATUS.json').read_text())
sh(f'tail -n 5 {RUN}/log.txt; ls -la {RUN}/ckpt {RUN}/checks', check=False)'''),
]

NB03 = [
    md("# 03-eval — DỰ PHÒNG: dự đoán test + đánh giá trên Kaggle\n\n"
       "Chỉ dùng khi máy nhà không chạy được (require 5.4, ghi nhật ký thay đổi). Test chạy **đúng một lần** (MR-2).\n"
       "Accelerator **GPU T4 x2**, commit. Input: `dsp-code`, `dsp-ag-data`, `dsp-ag-annotations`, `dsp-a1-state` (có `epoch4_weights.pth`)."),
    code(CELL_T0), code(CELL_CODE), code(CELL_INPUTS),
    code(r'''# Trọng số và JSON test
W = next(w for p in mounted() if p.name not in BIG_INPUTS for w in p.rglob('epoch4_weights.pth'))
st = json.loads((W.parent.parent / 'STATUS.json').read_text())
sys.path.insert(0, str(TOOLS)); import ag_common
assert ag_common.sha256_file(W) == st['weights_sha256']['epoch4_weights.pth'], 'sha256 epoch4_weights.pth sai'
RUN = WORK / 'runs' / 'A1'; RUN.mkdir(parents=True, exist_ok=True)
for f in ('STATUS.json', 'sessions.csv', 'A1.yaml'):
    if (W.parent.parent / f).exists(): shutil.copy2(W.parent.parent / f, RUN / f)
RELTR_T = TMP / 'data' / 'reltr'
sh(f'python {TOOLS}/convert_ag.py --ann_dir {ANN} --frames_dir {FRAMES} --out_dir {RELTR_T} --splits test dev --no_size_check')'''),
    code(r'''# Dự đoán test (2 GPU, theo phần, có hạn giờ), gộp, chấm, báo cáo
sh(f'torchrun --nproc_per_node=2 {TOOLS}/predict_ag.py --ckpt {W} --ann_path {RELTR_T} --img_folder {FRAMES} --split test '
   f'--out_dir {RUN}/pred_parts --deadline_unix {int(T0 + 10*3600)}')
sh(f'python {TOOLS}/predict_ag.py --merge {RUN}/pred_parts --split test --ann_path {RELTR_T} --out {RUN}/pred_test.pt')
sh(f'python {TOOLS}/ag_eval.py --pred {RUN}/pred_test.pt --ann_dir {ANN} --split test --gt full filtersmall '
   f'--modes with semi no --iou 0.5 0.3 --out {RUN}/eval --workers {os.cpu_count()}')
sh(f'python {TOOLS}/make_report.py --run {RUN} --ann_dir {ANN} --img_folder {FRAMES}')
print((RUN / 'eval' / 'metrics.md').read_text())'''),
]


def to_ipynb(cells, path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(cells):
        lines = src.strip("\n").split("\n")
        c = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            c.update({"execution_count": None, "outputs": []})
        nb["cells"].append(c)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, cells in (("00-prep", NB00), ("01-checks", NB01), ("02-train", NB02), ("03-eval", NB03)):
        to_ipynb(cells, OUT / f"{name}.ipynb")
        print("ghi", OUT / f"{name}.ipynb", len(cells), "cell")
