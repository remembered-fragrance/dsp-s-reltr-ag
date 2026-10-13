"""G2 phần máy nhà (MR-12): so tập test AG-full / AG-filtersmall (bộ lọc STTran) với ag_test_coco_style.json của OED.

  python tools/g2_oed_match.py --ann_dir data/ag/annotations --oed_json data/oed/ag_test_coco_style.json --out checks/g2_oed_match.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--oed_json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.load(open(args.oed_json, encoding="utf-8"))
    oed_frames = {im["file_name"].split("frames/")[-1] for im in d["images"]}
    oed = {"videos": len(d.get("videos", [])) or len({f.split("/")[0] for f in oed_frames}), "frames": len(oed_frames),
           "objects": sum(1 for a in d["annotations"] if a["category_id"] != 1),
           "objects_short_edge_le_16px": sum(1 for a in d["annotations"] if a["category_id"] != 1 and min(a["bbox"][2:]) <= 16)}
    person = C.load_pkl(Path(args.ann_dir) / "person_bbox.pkl")
    res = {"oed": oed, "variants": {}}
    for v in ("full", "filtersmall"):
        obj = C.load_pkl(Path(args.ann_dir) / C.PKL[v])
        vids = C.sttran_filter(person, obj, "test")
        nv, nf, no = C.check_ref_counts(v, "test", vids, obj)
        fr = {f for fs in vids.values() for f in fs}
        res["variants"][v] = {"videos": nv, "frames": nf, "objects": no, "oed_minus_ours": len(oed_frames - fr),
                              "ours_minus_oed": len(fr - oed_frames),
                              "match": not (oed_frames ^ fr) and no == oed["objects"]}
    m = [v for v, r in res["variants"].items() if r["match"]]
    res["oed_matches"] = m
    res["decision_MR12"] = ("giữ AG-full là bảng chính" if m == ["full"] else
                            "OED khớp AG-filtersmall: đề xuất đổi chính/phụ, chờ người dùng" if m == ["filtersmall"] else
                            "không khớp biến thể nào: ghi lại, giữ AG-full, báo người dùng")
    res["pass"] = bool(m)
    res["time"] = C.now_str()
    C.dump_json(res, args.out, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
