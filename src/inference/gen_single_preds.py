import json
import torch
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO

from torchvision.models.detection import fasterrcnn_resnet50_fpn, fcos_resnet50_fpn
from torchvision.transforms import functional as F
from ultralytics import YOLO
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

# 1. Path Configuration (Relative paths for GitHub)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ANN_FILE = DATA_DIR / "annotations" / "test_annotations.json"
IMG_DIR = DATA_DIR / "test_images"

# Output directory for single model predictions
OUT_DIR = BASE_DIR / "runs" / "single_preds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load COCO Ground Truth
if not ANN_FILE.exists():
    print(f"Error: Annotation file not found at {ANN_FILE}")
    exit(1)

coco_gt = COCO(str(ANN_FILE))
img_ids = coco_gt.getImgIds()

# ==========================================
# 2. Torchvision Models (FRCNN, FCOS)
# ==========================================
def run_torchvision_model(model_name="FRCNN"):
    """Inference for Faster R-CNN or FCOS using Torchvision weights."""
    print(f"Inference started: {model_name} (Device: {device})")
    if model_name == "FRCNN":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device)
        file_name = "pred_FRCNN_tv.json"
    else:
        model = fcos_resnet50_fpn(weights="DEFAULT").to(device)
        file_name = "pred_FCOS_tv.json"
    
    model.eval()
    results = []
    with torch.no_grad():
        for img_id in tqdm(img_ids, desc=f"Processing {model_name}"):
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = IMG_DIR / img_info['file_name']
            if not img_path.exists(): continue
            
            img = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(img).to(device)
            
            outputs = model([img_tensor])[0]
            for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1], # xyxy to xywh
                    "score": float(score)
                })
    
    with open(OUT_DIR / file_name, 'w') as f:
        json.dump(results, f)
    print(f"Successfully saved to: {OUT_DIR / file_name}")

# ==========================================
# 3. YOLOv8 (Direct JSON Construction)
# ==========================================
def run_yolov8():
    """Inference for YOLOv8 using Direct JSON Construction."""
    print("Inference started: YOLOv8")
    model = YOLO("yolov8n.pt") 
    
    results_list = []
    filename_to_id = {img['file_name']: img['id'] for img in coco_gt.loadImgs(img_ids)}
    
    # Process images as a stream
    preds = model.predict(source=str(IMG_DIR), conf=0.001, device=device, stream=True)
    
    for r in tqdm(preds, total=len(img_ids), desc="Processing YOLOv8"):
        file_name = os.path.basename(r.path)
        img_id = filename_to_id.get(file_name)
        
        if img_id is None: continue

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            results_list.append({
                "image_id": int(img_id),
                "category_id": int(label) + 1, # Convert 0-indexed to COCO 1-indexed
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            })

    target_json = OUT_DIR / "pred_YOLOv8_test.json"
    with open(target_json, 'w') as f:
        json.dump(results_list, f)
    print(f"Successfully saved YOLOv8 results to: {target_json}")

# ==========================================
# 4. Deformable DETR (HuggingFace)
# ==========================================
def run_deformable_detr():
    """Inference for Transformer-based Deformable DETR."""
    print(f"Inference started: Deformable DETR (Device: {device})")
    processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").to(device)
    model.eval()
    
    results = []
    with torch.no_grad():
        for img_id in tqdm(img_ids, desc="Processing DETR"):
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = IMG_DIR / img_info['file_name']
            if not img_path.exists(): continue
            
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            
            outputs = model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]]).to(device)
            processed = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.001)[0]
            
            for box, score, label in zip(processed['boxes'], processed['scores'], processed['labels']):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score)
                })
                
    save_path = OUT_DIR / "pred_DeformableDETR_hf.json"
    with open(save_path, 'w') as f:
        json.dump(results, f)
    print(f"Successfully saved to: {save_path}")

# ==========================================
# 5. Main Execution Logic
# ==========================================
if __name__ == "__main__":
    # Ensure all predictions are generated
    run_torchvision_model("FRCNN")
    run_torchvision_model("FCOS")
    run_yolov8()
    run_deformable_detr()
    print(f"\nAll model predictions are saved in: {OUT_DIR}")