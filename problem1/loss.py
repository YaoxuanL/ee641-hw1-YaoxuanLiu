import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

    def hard_negative_mining(self, loss, pos_mask, neg_mask, ratio=3):
        with torch.no_grad():
            num_pos = pos_mask.sum().item()
            num_neg_keep = int(ratio * max(1, num_pos))
            neg_loss = loss.clone()
            neg_loss[~neg_mask] = -1e9
            if neg_mask.any():
                _, idx = torch.topk(neg_loss, k=min(num_neg_keep, int(neg_mask.sum().item())), largest=True)
                selected = torch.zeros_like(neg_mask)
                selected[idx] = True
            else:
                selected = torch.zeros_like(neg_mask)
        return selected

    def forward(self, predictions, targets, anchors, num_classes=None):
        if num_classes is None:
            num_classes = self.num_classes

        B = predictions[0].shape[0]
        device = predictions[0].device
        total_loc = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)
        eps = 1e-6

        for si, pred in enumerate(predictions):
            _, ch, H, W = pred.shape
            A = anchors[si].shape[0] // (H*W)
            pred = pred.permute(0,2,3,1).contiguous().view(B, H*W*A, 5+num_classes)

            anc = anchors[si].to(device)
            for b in range(B):
                gt_boxes = targets[b]["boxes"].to(device)
                gt_labels = targets[b]["labels"].to(device)
                from utils import match_anchors_to_targets
                m_labels, m_boxes, pos_mask, neg_mask = match_anchors_to_targets(anc, gt_boxes, gt_labels)

                p = pred[b]
                p_tx, p_ty, p_tw, p_th = p[:,0], p[:,1], p[:,2], p[:,3]
                p_obj = p[:,4]
                p_cls = p[:,5:]

                ax = (anc[:,0] + anc[:,2]) * 0.5
                ay = (anc[:,1] + anc[:,3]) * 0.5
                aw = (anc[:,2] - anc[:,0]).clamp(min=eps)
                ah = (anc[:,3] - anc[:,1]).clamp(min=eps)

                gx = (m_boxes[:,0] + m_boxes[:,2]) * 0.5
                gy = (m_boxes[:,1] + m_boxes[:,3]) * 0.5
                gw = (m_boxes[:,2] - m_boxes[:,0]).clamp(min=eps)
                gh = (m_boxes[:,3] - m_boxes[:,1]).clamp(min=eps)

                t_tx = (gx - ax) / aw
                t_ty = (gy - ay) / ah
                t_tw = torch.log(gw / aw)
                t_th = torch.log(gh / ah)

                loc = F.smooth_l1_loss(p_tx[pos_mask], t_tx[pos_mask], reduction='sum')
                loc += F.smooth_l1_loss(p_ty[pos_mask], t_ty[pos_mask], reduction='sum')
                loc += F.smooth_l1_loss(p_tw[pos_mask], t_tw[pos_mask], reduction='sum')
                loc += F.smooth_l1_loss(p_th[pos_mask], t_th[pos_mask], reduction='sum')
                total_loc += loc

                obj_target = torch.zeros_like(p_obj)
                obj_target[pos_mask] = 1.0
                obj_loss_all = F.binary_cross_entropy_with_logits(p_obj, obj_target, reduction='none')
                neg_selected = self.hard_negative_mining(obj_loss_all.detach(), pos_mask, neg_mask, ratio=3)
                obj_loss = obj_loss_all[pos_mask].sum() + obj_loss_all[neg_selected].sum()
                total_obj += obj_loss

                if num_classes > 1:
                    cls_target = m_labels[pos_mask].clamp(min=0)
                    if cls_target.numel() > 0:
                        total_cls += F.cross_entropy(p_cls[pos_mask], cls_target, reduction='sum')

        N = max(1.0, float(B))
        loss = (total_loc + total_obj + total_cls) / N
        return loss, {"loss_loc": total_loc.item()/N, "loss_obj": total_obj.item()/N, "loss_cls": total_cls.item()/N}
