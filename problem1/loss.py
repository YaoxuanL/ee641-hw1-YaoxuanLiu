import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_anchors_to_targets

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.w_obj = 1.0
        self.w_cls = 1.0
        self.w_loc = 2.0
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.smoothl1 = nn.SmoothL1Loss(reduction="none")

    def forward(self, predictions, targets, anchors):
        B = predictions[0].shape[0]
        loss_obj_sum = 0.0
        loss_cls_sum = 0.0
        loss_loc_sum = 0.0
        for si, pred in enumerate(predictions):
            B, C, H, W = pred.shape
            A = anchors[si].shape[0] // (H * W)
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H * W * A, 5 + self.num_classes)
            anc = anchors[si].to(pred.device)
            ax = (anc[:, 0] + anc[:, 2]) * 0.5
            ay = (anc[:, 1] + anc[:, 3]) * 0.5
            aw = (anc[:, 2] - anc[:, 0]).clamp(min=1e-6)
            ah = (anc[:, 3] - anc[:, 1]).clamp(min=1e-6)
            for b in range(B):
                t = targets[b]
                boxes_gt = t["boxes"].to(pred.device)
                labels_gt = t["labels"].to(pred.device)
                matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(
                    anc, boxes_gt, labels_gt, pos_threshold=0.5, neg_threshold=0.3
                )
                matched_labels = matched_labels.long()
                pos_mask = pos_mask.bool()
                neg_mask = neg_mask.bool()
                p = pred[b]
                obj_logit = p[:, 4]
                cls_logit = p[:, 5:]
                tx, ty, tw, th = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
                obj_target = torch.zeros_like(obj_logit)
                obj_target[pos_mask] = 1.0
                obj_loss_all = self.bce(obj_logit, obj_target)
                sel_neg_mask = self.hard_negative_mining(
                    obj_loss_all.detach(), pos_mask, neg_mask, ratio=3
                )
                obj_use_mask = pos_mask | sel_neg_mask
                loss_obj = obj_loss_all[obj_use_mask].mean() if obj_use_mask.any() else obj_loss_all.new_tensor(0.0)
                if pos_mask.any():
                    cls_tgt = (matched_labels[pos_mask] - 1).clamp(min=0)
                    loss_cls = self.ce(cls_logit[pos_mask], cls_tgt).mean()
                else:
                    loss_cls = cls_logit.new_tensor(0.0)
                if pos_mask.any():
                    mb = matched_boxes[pos_mask]
                    gx = (mb[:, 0] + mb[:, 2]) * 0.5
                    gy = (mb[:, 1] + mb[:, 3]) * 0.5
                    gw = (mb[:, 2] - mb[:, 0]).clamp(min=1e-6)
                    gh = (mb[:, 3] - mb[:, 1]).clamp(min=1e-6)
                    ax_pos = ax[pos_mask]
                    ay_pos = ay[pos_mask]
                    aw_pos = aw[pos_mask]
                    ah_pos = ah[pos_mask]
                    t_tx = (gx - ax_pos) / aw_pos
                    t_ty = (gy - ay_pos) / ah_pos
                    t_tw = torch.log(gw / aw_pos)
                    t_th = torch.log(gh / ah_pos)
                    loc_pred = torch.stack([tx[pos_mask], ty[pos_mask], tw[pos_mask], th[pos_mask]], dim=1)
                    loc_tgt = torch.stack([t_tx, t_ty, t_tw, t_th], dim=1)
                    loss_loc = self.smoothl1(loc_pred, loc_tgt).mean()
                else:
                    loss_loc = obj_logit.new_tensor(0.0)
                loss_obj_sum += loss_obj
                loss_cls_sum += loss_cls
                loss_loc_sum += loss_loc
        loss_obj_mean = loss_obj_sum / max(1.0, float(B))
        loss_cls_mean = loss_cls_sum / max(1.0, float(B))
        loss_loc_mean = loss_loc_sum / max(1.0, float(B))
        loss_total = self.w_obj * loss_obj_mean + self.w_cls * loss_cls_mean + self.w_loc * loss_loc_mean
        loss_dict = {
            "loss_obj": loss_obj_mean.detach(),
            "loss_cls": loss_cls_mean.detach(),
            "loss_loc": loss_loc_mean.detach(),
            "loss_total": loss_total.detach(),
        }
        return loss_total, loss_dict

    def hard_negative_mining(self, loss, pos_mask, neg_mask, ratio=3):
        num_pos = int(pos_mask.sum().item())
        neg_candidates = loss.clone()
        neg_candidates[~neg_mask] = -1.0
        k = max(0, min(int(ratio * num_pos), int(neg_mask.sum().item())))
        if k == 0:
            return torch.zeros_like(neg_mask, dtype=torch.bool)
        vals, idx = torch.topk(neg_candidates, k=k, dim=0)
        selected = torch.zeros_like(neg_mask, dtype=torch.bool)
        selected[idx] = True
        return selected
