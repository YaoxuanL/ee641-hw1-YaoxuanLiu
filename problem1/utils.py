import torch

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    all_anchors = []
    for (H, W), scales in zip(feature_map_sizes, anchor_scales):
        anchors_level = []
        for i in range(H):
            for j in range(W):
                cx = (j + 0.5) * (image_size / W)
                cy = (i + 0.5) * (image_size / H)
                for s in scales:
                    side = s * image_size
                    x1 = cx - side/2.0
                    y1 = cy - side/2.0
                    x2 = cx + side/2.0
                    y2 = cy + side/2.0
                    anchors_level.append([x1, y1, x2, y2])
        all_anchors.append(torch.tensor(anchors_level, dtype=torch.float32))
    return all_anchors

def compute_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32, device=boxes1.device)
    b1 = boxes1[:, None, :]
    b2 = boxes2[None, :, :]
    x1 = torch.maximum(b1[...,0], b2[...,0])
    y1 = torch.maximum(b1[...,1], b2[...,1])
    x2 = torch.minimum(b1[...,2], b2[...,2])
    y2 = torch.minimum(b1[...,3], b2[...,3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0) * (boxes1[:,3]-boxes1[:,1]).clamp(min=0)
    a2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0) * (boxes2[:,3]-boxes2[:,1]).clamp(min=0)
    union = a1[:,None] + a2[None,:] - inter
    iou = torch.where(union>0, inter/union, torch.zeros_like(union))
    return iou

def match_anchors_to_targets(anchors, target_boxes, target_labels, pos_threshold=0.5, neg_threshold=0.3):
    K = anchors.shape[0]
    device = anchors.device
    if target_boxes.numel() == 0:
        matched_labels = torch.zeros((K,), dtype=torch.long, device=device)
        matched_boxes = torch.zeros((K,4), dtype=torch.float32, device=device)
        pos_mask = torch.zeros((K,), dtype=torch.bool, device=device)
        neg_mask = torch.ones((K,), dtype=torch.bool, device=device)
        return matched_labels, matched_boxes, pos_mask, neg_mask

    iou = compute_iou(anchors, target_boxes.to(device))
    best_iou, best_idx = iou.max(dim=1)
    matched_boxes = target_boxes[best_idx].to(device)
    matched_labels = target_labels[best_idx].to(device)

    pos_mask = best_iou >= pos_threshold
    neg_mask = best_iou < neg_threshold
    matched_labels = torch.where(pos_mask, matched_labels, torch.zeros_like(matched_labels))
    return matched_labels, matched_boxes, pos_mask, neg_mask
