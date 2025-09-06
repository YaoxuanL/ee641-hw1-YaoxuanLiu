import os, torch, json, random
from torch.utils.data import DataLoader, TensorDataset
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from evaluate import extract_keypoints_from_heatmaps, compute_pck, plot_pck_curves, visualize_predictions
import matplotlib.pyplot as plt

os.makedirs('results/visualizations/samples', exist_ok=True)

val_img = '/content/datasets/keypoints/val'
val_ann = '/content/datasets/keypoints/val_annotations.json'

ds_h = KeypointDataset(val_img, val_ann, output_type='heatmap', heatmap_size=64, sigma=2.0)
ds_r = KeypointDataset(val_img, val_ann, output_type='regression', heatmap_size=64, sigma=2.0)
dl_h = DataLoader(ds_h, batch_size=64, shuffle=False, num_workers=0)
dl_r = DataLoader(ds_r, batch_size=64, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hm = HeatmapNet().to(device)
rg = RegressionNet().to(device)
hm.load_state_dict(torch.load('results/heatmap_model.pth', map_location=device))
rg.load_state_dict(torch.load('results/regression_model.pth', map_location=device))
hm.eval(); rg.eval()

preds_h, gts_h = [], []
with torch.no_grad():
    for x, yhm in dl_h:
        x = x.to(device)
        ph = extract_keypoints_from_heatmaps(hm(x))
        gh = extract_keypoints_from_heatmaps(yhm.to(device))
        preds_h.append(ph.cpu()); gts_h.append(gh.cpu())
preds_h = torch.cat(preds_h, 0); gts_h = torch.cat(gts_h, 0)

preds_r = []
with torch.no_grad():
    for x, _ in dl_r:
        x = x.to(device)
        pr = rg(x).view(x.size(0), -1, 2)
        pr_64 = pr * 64.0
        preds_r.append(pr_64.cpu())
preds_r = torch.cat(preds_r, 0)

ths = [i/100.0 for i in range(2, 21, 2)]
pck_h = compute_pck(preds_h, gts_h, ths, normalize_by='bbox')
pck_r = compute_pck(preds_r, gts_h, ths, normalize_by='bbox')
plot_pck_curves(pck_h, pck_r, 'results/visualizations/pck_curve.png')
with open('results/visualizations/pck.json','w') as f:
    json.dump({'heatmap':pck_h, 'regression':pck_r}, f)

idxs = random.sample(range(len(ds_h)), min(12, len(ds_h)))
for i, idx in enumerate(idxs):
    img_h, _ = ds_h[idx]
    ph = preds_h[idx]
    gh = gts_h[idx]
    visualize_predictions(img_h, ph, gh, f'results/visualizations/samples/heatmap_pred_{i}.png')

for i, idx in enumerate(idxs):
    img_r, _ = ds_r[idx]
    pr = preds_r[idx]
    gh = gts_h[idx]
    visualize_predictions(img_r, pr, gh, f'results/visualizations/samples/regression_pred_{i}.png')

stages = [p for p in ['results/heatmap_epoch_10.pth','results/heatmap_epoch_20.pth','results/heatmap_epoch_30.pth'] if os.path.exists(p)]
if stages:
    x_vis, _ = next(iter(DataLoader(ds_h, batch_size=8, shuffle=False)))
    x_vis = x_vis.to(device)
    with torch.no_grad():
        for ckpt in stages:
            tag = os.path.splitext(os.path.basename(ckpt))[0]
            m = HeatmapNet().to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            hms = m(x_vis)
            for j in range(min(8, hms.size(0))):
                hm_img = hms[j].sum(0).cpu().numpy()
                plt.figure()
                plt.imshow(hm_img)
                plt.axis('off')
                plt.savefig(f'results/visualizations/{tag}_sample{j}.png', bbox_inches='tight', pad_inches=0)
                plt.close()

imgs = []
for i in range(len(ds_h)):
    img, _ = ds_h[i]
    imgs.append(img)
images = torch.stack(imgs, 0)
coords = gts_h.float()
test_loader = DataLoader(TensorDataset(images, coords), batch_size=64, shuffle=False, num_workers=0)

try:
    from baseline import analyze_failure_cases
    out = analyze_failure_cases({'heatmap':hm, 'regression':rg}, test_loader)
    with open('results/failure_cases.json','w') as f:
        json.dump(out, f)
except Exception as e:
    with open('results/failure_cases.json','w') as f:
        json.dump({'error': str(e)}, f)
