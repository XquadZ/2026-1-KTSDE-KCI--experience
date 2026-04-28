import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class EnsembleMetrics:
    def __init__(self, annotation_file):
        """Initialize with COCO Ground Truth."""
        self.coco_gt = COCO(annotation_file)
        self.cat_ids = self.coco_gt.getCatIds()

    def get_error_vectors(self, pred_json):
        """Identify detection status for each GT object (0: Detected, 1: Missed)."""
        coco_dt = self.coco_gt.loadRes(pred_json)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        gt_status = []
        # Extract matching results from COCO eval internal object
        for img_eval in coco_eval.evalImgs:
            if img_eval is None: continue
            # dtIg: detection ignored, gtIg: ground truth ignored
            # gtMatches: IDs of detections matched to each GT (0 if missed)
            matches = img_eval['gtMatches'][0] 
            for m in matches:
                gt_status.append(0 if m > 0 else 1)
                
        return np.array(gt_status)

    @staticmethod
    def calculate_gain_miss(vec_a, vec_b):
        """Equation (5): Gain of Missed Detection."""
        miss_a = np.mean(vec_a)
        miss_b = np.mean(vec_b)
        joint_miss = np.mean(vec_a * vec_b)
        return min(miss_a, miss_b) - joint_miss

    @staticmethod
    def calculate_disagreement(vec_a, vec_b):
        """Equation (3): Disagreement Rate."""
        n10 = np.sum((vec_a == 1) & (vec_b == 0))
        n01 = np.sum((vec_a == 0) & (vec_b == 1))
        return (n10 + n01) / len(vec_a)

    @staticmethod
    def calculate_ufp_risk(preds_a, preds_b):
        """Equation (6): Unique False Positive Risk."""
        # Identification of unique FP boxes using image_id and bbox overlap
        set_a = {(p['image_id'], tuple(p['bbox'])) for p in preds_a}
        set_b = {(p['image_id'], tuple(p['bbox'])) for p in preds_b}
        
        union = len(set_a | set_b)
        if union == 0: return 0.0
        
        intersection = len(set_a & set_b)
        unique_fps = union - intersection
        return unique_fps / union