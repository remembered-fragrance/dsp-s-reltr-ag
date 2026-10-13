"""Định dạng dự đoán dùng chung cho predict_ag.py, ag_eval.py, P13 (dev trong lúc train) và demo.

Mỗi frame lưu 2 tensor float16 (require FR-5):
  'trip': (Q<=100, 37) — top-Q query bộ ba theo p(person) × max p(vật):
          [0:4] box subject xyxy (pixel ảnh gốc), [4:8] box object xyxy, [8] p(person),
          [9] lớp vật (2..36), [10] điểm vật, [11:37] xác suất quan hệ theo chỉ số STTran 0..25,
          đã chuẩn hóa RIÊNG từng nhóm (attention 0–2, spatial 3–8, contacting 9–25) ở float32 trước khi đổi sang float16.
  'ent':  (100, 6) — phát hiện của entity decoder: [0:4] box xyxy, [4] lớp, [5] điểm (hậu xử lý kiểu DETR).

torch chỉ import trong hàm: worker của ag_eval.py (spawn trên Windows) nạp lại module này mà không cần torch;
mỗi tiến trình nạp torch cu128 tốn khoảng 1GB bộ nhớ commit, 15 worker làm Windows báo MemoryError.
"""
TOPQ = 100
C_SUB, C_OBJ, C_PP, C_OL, C_OS, C_REL = slice(0, 4), slice(4, 8), 8, 9, 10, slice(11, 37)
N_TRIP_COLS = 37
GROUP_SLICES = (slice(0, 3), slice(3, 9), slice(9, 26))
PERSON = 1


def _cxcywh_to_xyxy_px(b, w, h):
    import torch
    cx, cy, bw, bh = b.unbind(-1)
    out = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
    return out * torch.tensor([w, h, w, h], dtype=out.dtype, device=out.device)


def pack_outputs(outputs, sizes_wh, topq=TOPQ):
    """outputs: dict của RelTR (batch); sizes_wh: list (w, h) ảnh gốc. Trả về list dict {'trip','ent'} (float16, CPU)."""
    import torch
    with torch.no_grad():
        return _pack(torch, outputs, sizes_wh, topq)


def _pack(torch, outputs, sizes_wh, topq):
    res = []
    for b, (w, h) in enumerate(sizes_wh):
        sub_p = outputs["sub_logits"][b].float().softmax(-1)          # (T, 38): 0..36 + no-object
        obj_p = outputs["obj_logits"][b].float().softmax(-1)
        p_person = sub_p[:, PERSON]
        obj_score, obj_label = obj_p[:, 2:-1].max(-1)                  # lớp vật 2..36
        obj_label = obj_label + 2
        rel_logits = outputs["rel_logits"][b].float()[:, 1:-1]        # bỏ lớp giả 0 và "không quan hệ"
        rel = torch.cat([rel_logits[:, s].softmax(-1) for s in GROUP_SLICES], -1)
        order = (p_person * obj_score).argsort(descending=True)[:topq]
        trip = torch.cat([
            _cxcywh_to_xyxy_px(outputs["sub_boxes"][b].float(), w, h),
            _cxcywh_to_xyxy_px(outputs["obj_boxes"][b].float(), w, h),
            p_person[:, None], obj_label[:, None].float(), obj_score[:, None], rel], -1)[order]
        ent_p = outputs["pred_logits"][b].float().softmax(-1)
        ent_score, ent_label = ent_p[:, :-1].max(-1)
        ent = torch.cat([_cxcywh_to_xyxy_px(outputs["pred_boxes"][b].float(), w, h),
                         ent_label[:, None].float(), ent_score[:, None]], -1)
        res.append({"trip": trip.half().cpu(), "ent": ent.half().cpu()})
    return res
