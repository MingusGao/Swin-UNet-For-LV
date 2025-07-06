import torch
import numpy as np
import cv2


def seg_to_area(mask: torch.Tensor or np.ndarray) -> float:

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    return np.sum(mask)


def get_long_axis_length(mask: torch.Tensor or np.ndarray) -> float:

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea)


    hull = cv2.convexHull(contour)


    max_dist = 0
    p1, p2 = None, None

    for i in range(hull.shape[0]):
        for j in range(i + 1, hull.shape[0]):
            dist = np.linalg.norm(hull[i] - hull[j])
            if dist > max_dist:
                max_dist = dist
                p1 = hull[i]
                p2 = hull[j]

    return float(max_dist) if p1 is not None else 0.0

def calculate_ef(area_ed: float, area_es: float) -> float:

    if area_ed <= 0:
        return 0.0

    ef = ((area_ed - area_es) / area_ed) * 100.0
    return max(0.0, min(ef, 100.0))


def calculate_ef_from_volumes(edv: float, esv: float) -> float:

    if edv <= 0:
        return 0.0

    ef = ((edv - esv) / edv) * 100.0
    return max(0.0, min(ef, 100.0))


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, mask_available: torch.Tensor, smooth: float = 1e-6):

    
    logits = logits.squeeze(2)
    targets = targets.squeeze(2)


    valid_logits = logits[mask_available]
    valid_targets = targets[mask_available]


    if valid_logits.numel() == 0:
        return None, None


    preds = torch.sigmoid(valid_logits) > 0.5
    preds = preds.float()
    valid_targets = valid_targets.float()


    preds_flat = preds.view(preds.shape[0], -1)
    targets_flat = valid_targets.view(valid_targets.shape[0], -1)


    intersection = (preds_flat * targets_flat).sum(dim=1)
    total_sum = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    union = total_sum - intersection

 
    dice_score = (2. * intersection + smooth) / (total_sum + smooth)
    iou_score = (intersection + smooth) / (union + smooth)


    return dice_score.mean().item(), iou_score.mean().item()
