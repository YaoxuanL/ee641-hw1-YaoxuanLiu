import torch
import numpy as np
import matplotlib.pyplot as plt

def extract_keypoints_from_heatmaps(heatmaps):
    b, k, h, w = heatmaps.shape
    flat = heatmaps.view(b, k, -1)
    idx = flat.argmax(dim=2)
    y = (idx // w).float()
    x = (idx % w).float()
    return torch.stack([x, y], dim=2)

def compute_pck(predictions, ground_truths, thresholds, normalize_by='bbox'):
    preds = predictions.detach().cpu().float()
    gts = ground_truths.detach().cpu().float()
    N, K, _ = preds.shape
    if normalize_by == 'torso':
        norms = (gts[:,0,:] - gts[:,1,:]).norm(dim=1) + 1e-6
    else:
        x_min, _ = gts[:,:,0].min(dim=1)
        y_min, _ = gts[:,:,1].min(dim=1)
        x_max, _ = gts[:,:,0].max(dim=1)
        y_max, _ = gts[:,:,1].max(dim=1)
        norms = torch.sqrt((x_max - x_min)**2 + (y_max - y_min)**2) + 1e-6
    dists = (preds - gts).norm(dim=2)
    out = {}
    for t in thresholds:
        thr = norms.unsqueeze(1) * float(t)
        correct = (dists <= thr).sum().item()
        out[t] = correct / float(N * K)
    return out

def plot_pck_curves(pck_heatmap, pck_regression, save_path):
    ts = sorted(set(list(pck_heatmap.keys()) + list(pck_regression.keys())))
    y1 = [pck_heatmap.get(t, 0.0) for t in ts]
    y2 = [pck_regression.get(t, 0.0) for t in ts]
    plt.figure()
    plt.plot(ts, y1, label='heatmap')
    plt.plot(ts, y2, label='regression')
    plt.xlabel('threshold')
    plt.ylabel('PCK')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_predictions(image, pred_keypoints, gt_keypoints, save_path):
    img = image.squeeze(0).cpu().numpy()
    pk = pred_keypoints.detach().cpu().numpy()
    gk = gt_keypoints.detach().cpu().numpy()
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.scatter(pk[:,0], pk[:,1], marker='x')
    plt.scatter(gk[:,0], gk[:,1], marker='o')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
