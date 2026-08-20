import numpy as np

def calculate_bbox_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) for two bounding boxes.
    Boxes should be in format: [x1, y1, x2, y2]
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def calculate_temporal_iou(predicted_segment, truth_segment):
    """
    Calculates Temporal IoU for action phase detection (e.g., 'Impact Phase').
    Segments should be in format: (start_time_seconds, end_time_seconds)
    """
    pred_start, pred_end = predicted_segment
    gt_start, gt_end = truth_segment

    # Calculate intersection
    intersect_start = max(pred_start, gt_start)
    intersect_end = min(pred_end, gt_end)
    intersection = max(0, intersect_end - intersect_start)

    # Calculate union
    pred_duration = pred_end - pred_start
    gt_duration = gt_end - gt_start
    union = pred_duration + gt_duration - intersection

    if union == 0:
        return 0.0

    return intersection / union

if __name__ == "__main__":
    print("==================================================")
    print("🏏 POSTURE EXPERT: EVALUATION METRICS MODULE")
    print("==================================================")
    
    # --- 1. Test Bounding Box IoU ---
    # Imagine a human labeled the exact box, and our YOLO model predicted a slightly different one
    ground_truth_box = [100, 100, 200, 200]  # [x1, y1, x2, y2]
    predicted_box = [90, 110, 210, 190]
    
    bbox_iou = calculate_bbox_iou(ground_truth_box, predicted_box)
    
    print("\n[1] Spatial Accuracy (Bounding Box IoU)")
    print(f"Ground Truth Box: {ground_truth_box}")
    print(f"Predicted Box:    {predicted_box}")
    print(f"Calculated IoU:   {bbox_iou:.4f} ({(bbox_iou*100):.1f}%)")
    if bbox_iou > 0.5:
        print("Result: ✅ True Positive (IoU > 0.50)")
        
    # --- 2. Test Temporal IoU ---
    # Imagine the real "Downswing & Impact" phase happens from 2.5s to 3.0s
    # Our AI predicted it happened from 2.6s to 3.1s
    ground_truth_phase = (2.5, 3.0)
    predicted_phase = (2.6, 3.1)
    
    temp_iou = calculate_temporal_iou(predicted_phase, ground_truth_phase)
    
    print("\n[2] Phase Detection Accuracy (Temporal IoU)")
    print(f"Ground Truth 'Impact' Phase: {ground_truth_phase[0]}s to {ground_truth_phase[1]}s")
    print(f"Predicted 'Impact' Phase:    {predicted_phase[0]}s to {predicted_phase[1]}s")
    print(f"Calculated Temporal IoU:     {temp_iou:.4f} ({(temp_iou*100):.1f}%)")
    if temp_iou > 0.5:
        print("Result: ✅ Accurate Phase Detection (IoU > 0.50)")
        
    print("\n==================================================")
    print("💡 INTERVIEW TALKING POINT:")
    print("This module proves I know how to programmatically calculate")
    print("Intersection over Union (IoU) for both Spatial Object Detection")
    print("and Temporal Action Recognition.")
    print("==================================================")
