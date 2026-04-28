# -*- coding: utf-8 -*-
"""
Pre-ensemble selection experiment (single preds only) + verification on actual ensembles
- Computes pre-ensemble indicators (miss complementarity + FP risk) from single predictions
- Evaluates actual ensemble json files (NMS/WBF) using COCOeval
- Auto-fixes Category ID mismatches (YOLO 0-79 to COCO 1-90) and filters noise
"""

import os
import re
import json
import math
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =========================================================
# 0) CONFIG: Relative paths for GitHub repository
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"

ANN_FILE = DATA_DIR / "annotations" / "test_annotations.json"

SINGLE_PRED_PATHS = {
    "DeformableDETR": RUNS_DIR / "single_preds" / "pred_DeformableDETR_hf.json",
    "FRCNN":          RUNS_DIR / "single_preds" / "pred_FRCNN_tv.json",
    "FCOS":           RUNS_DIR / "single_preds" / "pred_FCOS_tv.json",
    "YOLOv8":         RUNS_DIR / "single_preds" / "pred_YOLOv8_test.json",
}

ENSEMBLE_DIRS = [
    RUNS_DIR / "pairwise_ens",
    RUNS_DIR / "combo_ens",
]

TARGET_MODELS = set(SINGLE_PRED_PATHS.keys())

# Evaluation constants
IOU_THR_MATCH = 0.50
FP_SCORE_THR  = 0.05
FP_IOU_THR    = 0.50
TFC_TOPK      = 50
TFC_SCORE_MIN = 0.00

# Output statistics directory
OUT_DIR = BASE_DIR / "outputs" / "stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.6f}".format)


# =========================================================
# DATA FIXER: Fixes Category IDs and removes noise
# =========================================================
COCO_MAPPING = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
                85, 86, 87, 88, 89, 90]

def fix_and_filter_predictions():
    """Corrects category IDs and filters low-confidence predictions."""
    print("\n[System] Fixing single model JSON files (mapping and noise filtering)...")
    for m, path in SINGLE_PRED_PATHS.items():
        if not Path(path).exists(): continue
        with open(path, 'r') as f:
            data = json.load(f)
        
        cat_ids = set(int(p.get('category_id', -1)) for p in data)
        needs_mapping = (0 in cat_ids and max(cat_ids) < 80)
        
        fixed_data = []
        for p in data:
            if float(p.get('score', 0)) < 0.05:
                continue
            
            cat_id = int(p.get('category_id', -1))
            if needs_mapping and 0 <= cat_id < 80:
                cat_id = COCO_MAPPING[cat_id]
            
            p['category_id'] = cat_id
            fixed_data.append(p)
        
        with open(path, 'w') as f:
            json.dump(fixed_data, f)
        print(f"  Fixed {m:14s}: {len(data)} -> {len(fixed_data)} boxes (Mapped: {needs_mapping})")


# =========================================================
# 1) Geometry + Helpers
# =========================================================
def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def box_area_xyxy(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = box_area_xyxy(a) + box_area_xyxy(b) - inter
    return inter / ua if ua > 0 else 0.0

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def ensure_exists(path, name="path"):
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} not found: {path}")

def scale_from_area(area):
    if area < 32.0 * 32.0: return "small"
    if area < 96.0 * 96.0: return "medium"
    return "large"

def norm_minmax(series: pd.Series):
    vmin = series.min()
    vmax = series.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - vmin) / (vmax - vmin)

# =========================================================
# 2) Load/Sanitize Predictions
# =========================================================
def load_pred_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Prediction json must be a list: {path}")
    return data

def sanitize_preds(preds, valid_img_ids, valid_cat_ids):
    out = []
    for p in preds:
        if "image_id" not in p or "category_id" not in p or "bbox" not in p: continue
        try:
            img_id = int(p["image_id"])
            cat_id = int(p["category_id"])
        except Exception: continue
        if img_id not in valid_img_ids or cat_id not in valid_cat_ids: continue
        bbox = p["bbox"]
        if (not isinstance(bbox, (list, tuple))) or len(bbox) != 4: continue
        score = safe_float(p.get("score", 1.0), default=1.0)
        bb = [safe_float(bbox[0]), safe_float(bbox[1]), safe_float(bbox[2]), safe_float(bbox[3])]
        if not all(np.isfinite(bb)) or bb[2] <= 0 or bb[3] <= 0: continue
        out.append({"image_id": img_id, "category_id": cat_id, "bbox": bb, "score": score})
    return out

def group_preds_by_img_cat(preds):
    by = defaultdict(lambda: defaultdict(list))
    for p in preds:
        by[p["image_id"]][p["category_id"]].append(p)
    return by

# =========================================================
# 3) COCOeval Wrappers
# =========================================================
def coco_eval_stats(coco_gt: COCO, pred_json_path: str):
    coco_dt = coco_gt.loadRes(str(pred_json_path))
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats
    if stats is None or len(stats) < 12:
        raise RuntimeError(f"COCOeval.stats invalid for: {pred_json_path}")
    return [float(x) for x in stats]

def stats_to_dict(stats):
    s = stats
    return {
        "AP": s[0], "AP50": s[1], "AP75": s[2],
        "AP_S": s[3], "AP_M": s[4], "AP_L": s[5],
        "AR": s[8], "AR_S": s[9], "AR_M": s[10], "AR_L": s[11],
    }

# =========================================================
# 4) Build GT Object List
# =========================================================
def build_gt_objects(coco_gt: COCO):
    img_ids = sorted(coco_gt.getImgIds())
    gt_objs = []
    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        if not ann_ids: continue
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            if ann.get("iscrowd", 0) == 1: continue
            cat_id = int(ann["category_id"])
            bb = xywh_to_xyxy(ann["bbox"])
            area = float(ann["area"])
            sc = scale_from_area(area)
            gt_objs.append({"img_id": int(img_id), "cat_id": cat_id, "bbox_xyxy": bb, "area": area, "scale": sc})
    return gt_objs

# =========================================================
# 5) Miss Vectors + Matched Boxes
# =========================================================
def compute_miss_and_matches(gt_objs, preds_by_model_imgcat, iou_thr=0.5):
    models = list(preds_by_model_imgcat.keys())
    N = len(gt_objs)
    miss = {m: np.ones(N, dtype=np.int32) for m in models}
    match_box = {m: [None] * N for m in models}

    for idx, g in enumerate(gt_objs):
        img_id, cat_id, gt_box = g["img_id"], g["cat_id"], g["bbox_xyxy"]
        for m in models:
            plist = preds_by_model_imgcat[m].get(img_id, {}).get(cat_id, [])
            best_iou, best_box = 0.0, None
            for p in plist:
                pb = xywh_to_xyxy(p["bbox"])
                iou = iou_xyxy(gt_box, pb)
                if iou > best_iou:
                    best_iou, best_box = iou, pb
            if best_iou >= iou_thr:
                miss[m][idx] = 0
                match_box[m][idx] = best_box
    return miss, match_box

# =========================================================
# 6) Pairwise Contingency Metrics
# =========================================================
def pair_counts(va, vb):
    N11 = int(np.sum((va == 1) & (vb == 1)))
    N00 = int(np.sum((va == 0) & (vb == 0)))
    N10 = int(np.sum((va == 1) & (vb == 0)))
    N01 = int(np.sum((va == 0) & (vb == 1)))
    return N11, N00, N10, N01

def phi_from_counts(N11, N00, N10, N01):
    num = (N11 * N00) - (N10 * N01)
    den = math.sqrt((N11 + N10) * (N11 + N01) * (N00 + N10) * (N00 + N01))
    return (num / den) if den > 0 else 0.0

def kappa_from_counts(N11, N00, N10, N01):
    N = N11 + N00 + N10 + N01
    if N == 0: return 0.0
    po = (N11 + N00) / N
    pe = ((N11 + N10) / N) * ((N11 + N01) / N) + ((N00 + N01) / N) * ((N00 + N10) / N)
    return (po - pe) / (1.0 - pe) if (1.0 - pe) > 1e-12 else 0.0

def yule_q_from_counts(N11, N00, N10, N01):
    ad, bc = N11 * N00, N10 * N01
    den = ad + bc
    return (ad - bc) / den if den > 0 else 0.0

def disagreement(va, vb):
    return float(np.mean(va != vb)) if len(va) else 0.0

# =========================================================
# 7) Risk Metrics: FP Sets (for UFP) + TFC
# =========================================================
def build_gt_by_img_cat(coco_gt: COCO, valid_img_ids, valid_cat_ids):
    gt = defaultdict(lambda: defaultdict(list))
    for img_id in valid_img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        if not ann_ids: continue
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            if ann.get("iscrowd", 0) == 1: continue
            cat = int(ann["category_id"])
            if cat not in valid_cat_ids: continue
            gt[img_id][cat].append(xywh_to_xyxy(ann["bbox"]))
    return gt

def greedy_tp_fp_for_image(pred_list, gt_boxes, iou_thr=0.5):
    used = [False] * len(gt_boxes)
    labels = []
    for pb, score in pred_list:
        best_j, best_iou = -1, 0.0
        for j, gb in enumerate(gt_boxes):
            if used[j]: continue
            iou = iou_xyxy(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            used[best_j] = True
            labels.append(True)
        else:
            labels.append(False)
    return labels

def compute_fp_set_for_model(preds, gt_by_img_cat, score_thr=0.05, iou_thr=0.5):
    by_img_cat = defaultdict(list)
    for p in preds:
        if p["score"] >= score_thr:
            by_img_cat[(p["image_id"], p["category_id"])].append(p)
    fp_items = []
    for (img_id, cat_id), plist in by_img_cat.items():
        plist_sorted = sorted(plist, key=lambda x: -x["score"])
        pred_boxes = [(xywh_to_xyxy(p["bbox"]), p["score"]) for p in plist_sorted]
        gt_boxes = gt_by_img_cat.get(img_id, {}).get(cat_id, [])
        labels = greedy_tp_fp_for_image(pred_boxes, gt_boxes, iou_thr=iou_thr)
        for (pb, _), is_tp in zip(pred_boxes, labels):
            if not is_tp: fp_items.append((img_id, cat_id, pb))
    return fp_items

def ufp_pair(fpA, fpB, iou_thr=0.5):
    A, B = defaultdict(list), defaultdict(list)
    for img, cat, bb in fpA: A[(img, cat)].append(bb)
    for img, cat, bb in fpB: B[(img, cat)].append(bb)
    overlap = 0
    totalA, totalB = len(fpA), len(fpB)
    for key in set(A.keys()) & set(B.keys()):
        a_list, b_list = A[key], B[key]
        used_b = [False] * len(b_list)
        for ab in a_list:
            best_j, best_iou = -1, 0.0
            for j, bb in enumerate(b_list):
                if used_b[j]: continue
                iou = iou_xyxy(ab, bb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                used_b[best_j] = True
                overlap += 1
    union = totalA + totalB - overlap
    unique = (totalA - overlap) + (totalB - overlap)
    return (unique / union) if union > 0 else 0.0

def tfc_model(preds, gt_by_img_cat, topk=50, score_min=0.0, iou_thr=0.5):
    preds_by_img = defaultdict(list)
    for p in preds:
        if p["score"] >= score_min: preds_by_img[p["image_id"]].append(p)
    ratios = []
    for img_id, plist in preds_by_img.items():
        plist_sorted = sorted(plist, key=lambda x: -x["score"])[:topk]
        if not plist_sorted: continue
        by_cat = defaultdict(list)
        for p in plist_sorted: by_cat[p["category_id"]].append(p)
        fp_count, total = 0, 0
        for cat_id, cplist in by_cat.items():
            cplist_sorted = sorted(cplist, key=lambda x: -x["score"])
            pred_boxes = [(xywh_to_xyxy(p["bbox"]), p["score"]) for p in cplist_sorted]
            gt_boxes = gt_by_img_cat.get(img_id, {}).get(cat_id, [])
            labels = greedy_tp_fp_for_image(pred_boxes, gt_boxes, iou_thr=iou_thr)
            for is_tp in labels:
                total += 1
                if not is_tp: fp_count += 1
        if total > 0: ratios.append(fp_count / total)
    return float(np.mean(ratios)) if ratios else 0.0

# =========================================================
# 8) Comp Metric
# =========================================================
def comp_pair(gt_idxs, match_box_A, match_box_B):
    ious = []
    for idx in gt_idxs:
        a, b = match_box_A[idx], match_box_B[idx]
        if a is not None and b is not None:
            ious.append(iou_xyxy(a, b))
    return float(np.mean(ious)) if ious else 0.0

# =========================================================
# 9) Pre-ensemble Indicators (Tables)
# =========================================================
def avg_pairwise_over_combo(models, func_pair):
    vals = [func_pair(a, b) for a, b in combinations(models, 2)]
    return float(np.mean(vals)) if vals else 0.0

def compute_preensemble_tables(gt_objs, miss, match_box, single_stats, preds_sanitized, coco_gt):
    models = list(miss.keys())
    idx_all = np.arange(len(gt_objs))
    idx_scale = {
        "all": idx_all,
        "small": np.array([i for i, g in enumerate(gt_objs) if g["scale"] == "small"], dtype=int),
        "medium": np.array([i for i, g in enumerate(gt_objs) if g["scale"] == "medium"], dtype=int),
        "large": np.array([i for i, g in enumerate(gt_objs) if g["scale"] == "large"], dtype=int),
    }

    valid_img_ids, valid_cat_ids = set(coco_gt.getImgIds()), set(coco_gt.getCatIds())
    gt_by_img_cat = build_gt_by_img_cat(coco_gt, valid_img_ids, valid_cat_ids)

    fp_sets, tfc_vals = {}, {}
    for m in models:
        fp_sets[m] = compute_fp_set_for_model(preds_sanitized[m], gt_by_img_cat, score_thr=FP_SCORE_THR, iou_thr=IOU_THR_MATCH)
        tfc_vals[m] = tfc_model(preds_sanitized[m], gt_by_img_cat, topk=TFC_TOPK, score_min=TFC_SCORE_MIN, iou_thr=IOU_THR_MATCH)

    rows = []
    for k in [2, 3, 4]:
        for combo in combinations(models, k):
            combo = tuple(combo)
            combo_name = "+".join(combo)
            for sc, idxs in idx_scale.items():
                if len(idxs) == 0: continue
                miss_rates, miss_vecs = [], []
                for m in combo:
                    v = miss[m][idxs]
                    miss_vecs.append(v)
                    miss_rates.append(float(np.mean(v)))
                
                and_miss = miss_vecs[0].copy()
                for v in miss_vecs[1:]: and_miss = and_miss & (v == 1)
                ensemble_miss = float(np.mean(and_miss))
                gain_vs_best = float(min(miss_rates)) - ensemble_miss

                avg_bias = float(np.mean(miss_rates))
                avg_variance = float(np.mean([(mr - avg_bias) ** 2 for mr in miss_rates]))
                
                def cov_pair(a, b):
                    va, vb = miss[a][idxs], miss[b][idxs]
                    return float(np.mean((va == 1) & (vb == 1))) - (float(np.mean(va)) * float(np.mean(vb)))
                covariance = avg_pairwise_over_combo(combo, cov_pair) if k > 1 else 0.0

                def metrics_pair(a, b):
                    va, vb = miss[a][idxs], miss[b][idxs]
                    miss_a, miss_b = float(np.mean(va)), float(np.mean(vb))
                    joint = float(np.mean((va == 1) & (vb == 1)))
                    N11, N00, N10, N01 = pair_counts(va, vb)
                    lift = (joint / (miss_a * miss_b)) if (miss_a * miss_b) > 1e-12 else 0.0
                    return lift, phi_from_counts(N11, N00, N10, N01), kappa_from_counts(N11, N00, N10, N01), yule_q_from_counts(N11, N00, N10, N01), disagreement(va, vb)

                lifts, phis, kappas, yules, diss = [], [], [], [], []
                for a, b in combinations(combo, 2):
                    lift, phi, kappa, yule, dis = metrics_pair(a, b)
                    lifts.append(lift); phis.append(phi); kappas.append(kappa); yules.append(yule); diss.append(dis)

                avg_lift = float(np.mean(lifts)) if lifts else 0.0
                avg_phi  = float(np.mean(phis)) if phis else 0.0
                avg_kappa = float(np.mean(kappas)) if kappas else 0.0
                avg_yule = float(np.mean(yules)) if yules else 0.0
                avg_dis = float(np.mean(diss)) if diss else 0.0

                avg_comp = avg_pairwise_over_combo(combo, lambda a, b: comp_pair(idxs, match_box[a], match_box[b]))
                avg_ufp = avg_pairwise_over_combo(combo, lambda a, b: ufp_pair(fp_sets[a], fp_sets[b], iou_thr=FP_IOU_THR))
                avg_tfc = float(np.mean([tfc_vals[m] for m in combo])) if combo else 0.0

                rows.append({"k": k, "Scale": sc, "Combo": combo_name, "Count_GT": int(len(idxs)), "Ensemble_Miss": ensemble_miss, "Gain_vs_Best": gain_vs_best, "Avg_Bias": avg_bias, "Avg_Variance": avg_variance, "Covariance": covariance, "Avg_Lift": avg_lift, "Avg_Phi": avg_phi, "Avg_Kappa": avg_kappa, "Avg_Yule_Q": avg_yule, "Avg_Disagreement": avg_dis, "Avg_Comp": avg_comp, "Avg_UFP": avg_ufp, "Avg_TFC": avg_tfc})

    df = pd.DataFrame(rows)
    df["S_raw"] = 0.0
    out_frames = []
    for (k, sc), sub in df.groupby(["k", "Scale"], sort=False):
        sub = sub.copy()
        sub["n_Gain"] = norm_minmax(sub["Gain_vs_Best"])
        sub["n_Dis"]  = norm_minmax(sub["Avg_Disagreement"])
        sub["n_UFP"]  = norm_minmax(sub["Avg_UFP"])
        sub["n_Comp"] = norm_minmax(sub["Avg_Comp"])
        sub["n_TFC"]  = norm_minmax(sub["Avg_TFC"])
        sub["S"] = sub["n_Gain"] + sub["n_Dis"] - sub["n_UFP"] + sub["n_Comp"] - sub["n_TFC"]
        out_frames.append(sub)

    df = pd.concat(out_frames, axis=0, ignore_index=True)
    cols = ["k", "Scale", "Combo", "Count_GT", "Gain_vs_Best", "Avg_Disagreement", "Avg_UFP", "Avg_Comp", "Avg_TFC", "Covariance", "Avg_Lift", "Avg_Phi", "Avg_Kappa", "Avg_Yule_Q", "S"]
    return df[cols].sort_values(["k", "Scale", "S"], ascending=[True, True, False]).reset_index(drop=True)

# =========================================================
# 10) Evaluate Actual Gains
# =========================================================
def parse_ensemble_filename(fname):
    base = Path(fname).stem
    tokens = base.split("_")
    if len(tokens) < 3: return None
    method = "NMS" if "NMS" in tokens else "WBF" if "WBF" in tokens else None
    if not method: return None
    tokens.remove(method)
    prefix = tokens[0].lower()
    if prefix not in ("pair", "combo"): return None
    model_tokens = [t for t in tokens[1:] if re.match(r"^[A-Za-z0-9]+$", t)]
    if len(model_tokens) < 2: return None
    models = tuple(model_tokens)
    return tuple(sorted(models)), method, len(models)

def evaluate_ensemble_files(coco_gt, single_stats, pre_df):
    """Calculates actual mAP/mAR gains for ensemble results."""
    rows = []
    pre_lookup = {(int(r["k"]), str(r["Scale"]), str(r["Combo"])): float(r["S"]) for _, r in pre_df.iterrows()}
    
    for d in ENSEMBLE_DIRS:
        dpath = Path(d)
        if not dpath.exists(): continue
        for fp in sorted(dpath.glob("*.json")):
            name = fp.name
            if name.startswith("summary_"): continue
            parsed = parse_ensemble_filename(name)
            if parsed is None: continue
            models_sorted, method, k = parsed
            if not set(models_sorted).issubset(TARGET_MODELS): continue
            combo_name = "+".join(models_sorted)
            
            try:
                st = coco_eval_stats(coco_gt, str(fp))
                st_d = stats_to_dict(st)
            except Exception as e: continue
            
            best = {key: max([single_stats[m][key] for m in models_sorted]) for key in ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AR", "AR_S", "AR_M", "AR_L"]}
            
            rows.append({
                "Dir": str(dpath), "File": name, "k": k, "Method": method, "Combo": combo_name,
                "EnAP": st_d["AP"], "EnAR": st_d["AR"], "GainAP": st_d["AP"] - best["AP"], "GainAR": st_d["AR"] - best["AR"],
                "EnAP50": st_d["AP50"], "EnAP75": st_d["AP75"],
                "EnAP_S": st_d["AP_S"], "EnAP_M": st_d["AP_M"], "EnAP_L": st_d["AP_L"],
                "GainAP_S": st_d["AP_S"] - best["AP_S"], "GainAP_M": st_d["AP_M"] - best["AP_M"], "GainAP_L": st_d["AP_L"] - best["AP_L"],
                "EnAR_S": st_d["AR_S"], "EnAR_M": st_d["AR_M"], "EnAR_L": st_d["AR_L"],
                "GainAR_S": st_d["AR_S"] - best["AR_S"], "GainAR_M": st_d["AR_M"] - best["AR_M"], "GainAR_L": st_d["AR_L"] - best["AR_L"],
                "S_all": pre_lookup.get((k, "all", combo_name), np.nan),
                "S_small": pre_lookup.get((k, "small", combo_name), np.nan),
                "S_medium": pre_lookup.get((k, "medium", combo_name), np.nan),
                "S_large": pre_lookup.get((k, "large", combo_name), np.nan),
            })
    df = pd.DataFrame(rows)
    if not df.empty: df = df.sort_values(["k", "Method", "GainAP"], ascending=[True, True, False]).reset_index(drop=True)
    return df

# =========================================================
# 11) Correlations
# =========================================================
def pearson_corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12: return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def spearman_corr(x, y):
    return pearson_corr(pd.Series(x).rank(method="average").to_numpy(dtype=float), pd.Series(y).rank(method="average").to_numpy(dtype=float))

def compute_correlations(merged_df):
    rows = []
    if merged_df.empty: return pd.DataFrame()
    for k in sorted(merged_df["k"].unique()):
        sub = merged_df[merged_df["k"] == k].copy()
        if sub.empty: continue
        for method in sorted(sub["Method"].unique()):
            sm = sub[sub["Method"] == method]
            if sm.empty: continue
            for sc, sapp, sap, sar in [("all", "S_all", "GainAP", "GainAR"), ("small", "S_small", "GainAP_S", "GainAR_S"), ("medium", "S_medium", "GainAP_M", "GainAR_M"), ("large", "S_large", "GainAP_L", "GainAR_L")]:
                rows.append({"k": k, "Method": method, "Scale": sc, "Metric": "S", "Spearman_vs_GainAP": spearman_corr(sm[sapp], sm[sap]), "Spearman_vs_GainAR": spearman_corr(sm[sapp], sm[sar])})
    return pd.DataFrame(rows)

# =========================================================
# 12) Selection Performance
# =========================================================
def selection_performance(pre_df, actual_df, single_stats):
    if pre_df.empty or actual_df.empty: return pd.DataFrame()
    pre_map = {(int(r["k"]), str(r["Scale"]), str(r["Combo"])): r.to_dict() for _, r in pre_df.iterrows()}
    rows = []
    for k in sorted(actual_df["k"].unique()):
        for method in sorted(actual_df["Method"].unique()):
            sub = actual_df[(actual_df["k"] == k) & (actual_df["Method"] == method)].copy()
            if sub.empty: continue
            for scale_key, gain_ap_col, gain_ar_col, S_col in [("all", "GainAP", "GainAR", "S_all"), ("small", "GainAP_S", "GainAR_S", "S_small"), ("medium", "GainAP_M", "GainAR_M", "S_medium"), ("large", "GainAP_L", "GainAR_L", "S_large")]:
                ss = sub.copy()[np.isfinite(sub[S_col])]
                if ss.empty: continue
                
                prop = ss.loc[ss[S_col].astype(float).idxmax()]
                cov_pick = ss.loc[ss["Combo"].apply(lambda c: float(pre_map.get((k, scale_key, c), {}).get("Covariance", np.nan))).astype(float).idxmin()]
                dis_pick = ss.loc[ss["Combo"].apply(lambda c: float(pre_map.get((k, scale_key, c), {}).get("Avg_Disagreement", np.nan))).astype(float).idxmax()]
                gm_pick = ss.loc[ss["Combo"].apply(lambda c: float(pre_map.get((k, scale_key, c), {}).get("Gain_vs_Best", np.nan))).astype(float).idxmax()]
                
                best_gain_ap, best_gain_ar = float(ss[gain_ap_col].max()), float(ss[gain_ar_col].max())
                
                def pack(label, picked):
                    ga, gr = float(picked[gain_ap_col]), float(picked[gain_ar_col])
                    return {"k": k, "Method": method, "Scale": scale_key, "Selector": label, "Selected_Combo": str(picked["Combo"]), "GainAP": ga, "GainAR": gr, "PositiveAP": int(ga > 0), "PositiveAR": int(gr > 0), "RegretAP": best_gain_ap - ga, "RegretAR": best_gain_ar - gr}
                
                rows.extend([pack("Proposed_S", prop), pack("Min_Covariance", cov_pick), pack("Max_Disagreement", dis_pick), pack("Max_Gain_miss", gm_pick)])
    df = pd.DataFrame(rows)
    if not df.empty: df = df.sort_values(["k", "Method", "Scale", "Selector"]).reset_index(drop=True)
    return df

# =========================================================
# 13) MAIN
# =========================================================
def main():
    ensure_exists(ANN_FILE, "ANN_FILE")
    for k, p in SINGLE_PRED_PATHS.items(): ensure_exists(p, f"SINGLE_PRED_PATHS[{k}]")

    # [NEW] Data correction and filtering
    fix_and_filter_predictions()

    print("\nLoading COCO GT...")
    coco_gt = COCO(str(ANN_FILE))
    valid_img_ids, valid_cat_ids = set(coco_gt.getImgIds()), set(coco_gt.getCatIds())

    print("\nLoading single predictions...")
    preds_sanitized, preds_by_imgcat = {}, {}
    for m, path in SINGLE_PRED_PATHS.items():
        san = sanitize_preds(load_pred_json(path), valid_img_ids, valid_cat_ids)
        preds_sanitized[m], preds_by_imgcat[m] = san, group_preds_by_img_cat(san)
        print(f"  {m:14s} | dets={len(san):6d} | file={Path(path).name}")

    print("\nSingle model COCOeval (fills best single model baselines)...")
    single_stats = {}
    for m, path in SINGLE_PRED_PATHS.items():
        single_stats[m] = stats_to_dict(coco_eval_stats(coco_gt, str(path)))
        print(f"  {m:14s} | AP={single_stats[m]['AP']:.4f} | AR={single_stats[m]['AR']:.4f}")

    print("\nBuilding GT objects and miss vectors...")
    gt_objs = build_gt_objects(coco_gt)
    miss, match_box = compute_miss_and_matches(gt_objs, preds_by_imgcat, iou_thr=IOU_THR_MATCH)
    
    print("\nComputing Table 4 (pre-ensemble indicators and Score S)...")
    pre_df = compute_preensemble_tables(gt_objs, miss, match_box, single_stats, preds_sanitized, coco_gt)
    pre_df.to_csv(OUT_DIR / "table4_preensemble_indicators.csv", index=False, encoding="utf-8-sig")

    print("\nEvaluating actual ensemble JSON files (Table 5)...")
    actual_df = evaluate_ensemble_files(coco_gt, single_stats, pre_df)
    actual_df.to_csv(OUT_DIR / "table5_actual_ensemble_gains.csv", index=False, encoding="utf-8-sig")

    if actual_df.empty:
        print("\nNo ensemble files evaluated. Check ENSEMBLE_DIRS.")
        return

    print("\nMerging pre and actual data...")
    actual_df.copy().to_csv(OUT_DIR / "merged_pre_vs_actual.csv", index=False, encoding="utf-8-sig")

    print("\nComputing Table 6 (correlation analysis)...")
    corr_df = compute_correlations(actual_df)
    corr_df.to_csv(OUT_DIR / "table6_correlations.csv", index=False, encoding="utf-8-sig")

    print("\nComputing Table 7 (selection performance analysis)...")
    sel_df = selection_performance(pre_df, actual_df, single_stats)
    sel_df.to_csv(OUT_DIR / "table7_selection_performance.csv", index=False, encoding="utf-8-sig")

    print("\nAll COCOeval and Metrics generation complete!")

if __name__ == "__main__":
    main()