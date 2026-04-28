import json
import os
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion, nms

# 1. Path and Model Configuration
# Use relative paths for better portability on GitHub
BASE_DIR = Path(__file__).resolve().parent
SINGLE_PRED_DIR = BASE_DIR / "runs" / "single_preds"
ANN_FILE = BASE_DIR / "data" / "annotations" / "test_annotations.json"

# Output directories for ensemble results
PAIR_OUT_DIR = BASE_DIR / "runs" / "pairwise_ens"
COMBO_OUT_DIR = BASE_DIR / "runs" / "combo_ens"

for d in [PAIR_OUT_DIR, COMBO_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "DeformableDETR": "pred_DeformableDETR_hf.json",
    "FRCNN": "pred_FRCNN_tv.json",
    "FCOS": "pred_FCOS_tv.json",
    "YOLOv8": "pred_YOLOv8_test.json"
}

MIN_SCORE_THRESH = 0.05 
# COCO standard category mapping (0-79 to 1-90)
COCO_MAPPING = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
                85, 86, 87, 88, 89, 90]

# 2. Metadata Helper
def get_image_metadata(coco_json):
    """Load image dimensions for box normalization."""
    with open(coco_json, 'r') as f:
        data = json.load(f)
    return {img['id']: (img['width'], img['height']) for img in data['images']}

print("Loading image metadata for normalization...")
if not ANN_FILE.exists():
    print(f"Error: Annotation file not found at {ANN_FILE}")
    exit(1)
IMG_METADATA = get_image_metadata(ANN_FILE)

# 3. Core Fusion Logic
def run_fusion_process(combo_models, method="WBF", iou_thr=0.50):
    combined_data = {} 
    
    for model_name in combo_models:
        path = SINGLE_PRED_DIR / MODEL_FILES[model_name]
        if not path.exists(): continue
            
        with open(path, 'r') as f:
            preds = json.load(f)
            
        # Detect if class mapping is required (e.g., for YOLO 0-indexed results)
        cat_ids = {int(p.get('category_id', -1)) for p in preds}
        needs_mapping = (0 in cat_ids)
            
        for p in preds:
            score = float(p.get('score', 0))
            if score < MIN_SCORE_THRESH: continue 
                
            img_id = p['image_id']
            if isinstance(img_id, str):
                img_id = int(Path(img_id).stem)

            cat_id = int(p['category_id'])
            if needs_mapping and 0 <= cat_id < 80:
                cat_id = COCO_MAPPING[cat_id]
                
            if img_id not in combined_data:
                combined_data[img_id] = {'boxes': [], 'scores': [], 'labels': []}
            
            w_img, h_img = IMG_METADATA[img_id]
            x, y, w, h = p['bbox']
            
            # Normalize boxes to [0, 1] range for ensemble-boxes library
            norm_box = [
                max(0, x / w_img),
                max(0, y / h_img),
                min(1, (x + w) / w_img),
                min(1, (y + h) / h_img)
            ]
            
            if (norm_box[2] <= norm_box[0]) or (norm_box[3] <= norm_box[1]): continue

            combined_data[img_id]['boxes'].append(norm_box)
            combined_data[img_id]['scores'].append(score)
            combined_data[img_id]['labels'].append(cat_id)

    final_results = []
    for img_id, data in tqdm(combined_data.items(), desc=f"{method}: {'+'.join(combo_models)}", leave=False):
        if not data['boxes']: continue
        
        boxes_list, scores_list, labels_list = [data['boxes']], [data['scores']], [data['labels']]
        weights = [1]

        # Apply box fusion algorithms (WBF or NMS)
        if method == "WBF":
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        else:
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)

        w_img, h_img = IMG_METADATA[img_id]
        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = b[0] * w_img, b[1] * h_img, b[2] * w_img, b[3] * h_img
            final_results.append({
                "image_id": int(img_id),
                "category_id": int(l),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s)
            })
    return final_results

# 4. Main Execution
def main():
    model_names = list(MODEL_FILES.keys())
    methods = ["WBF", "NMS"]
    
    for k in [2, 3]:
        for combo in combinations(model_names, k):
            combo_sorted = sorted(list(combo))
            prefix = "pair" if k == 2 else "combo"
            target_dir = PAIR_OUT_DIR if k == 2 else COMBO_OUT_DIR
            
            for method in methods:
                file_name = f"{prefix}_{'_'.join(combo_sorted)}_{method}.json"
                save_path = target_dir / file_name
                
                print(f"Processing: {file_name}")
                results = run_fusion_process(combo_sorted, method=method)
                
                with open(save_path, 'w') as f:
                    json.dump(results, f)

    print("\nFusion completed. Results saved in 'runs/' directory.")

if __name__ == "__main__":
    main()