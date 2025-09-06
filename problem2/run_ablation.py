import os, json
import matplotlib.pyplot as plt
from baseline import ablation_study

os.makedirs('results/visualizations', exist_ok=True)

train_img = '/content/datasets/keypoints/train'
val_img = '/content/datasets/keypoints/val'
train_ann = '/content/datasets/keypoints/train_annotations.json'
val_ann = '/content/datasets/keypoints/val_annotations.json'

# 改成直接传路径元组
res = ablation_study(
    {
        'train': (train_img, train_ann),
        'val': (val_img, val_ann)
    },
    None
)

with open('results/baseline_results.json','w') as f:
    json.dump(res, f, indent=2)

sizes = []
size_vals = []
for s in [32,64,128]:
    k = f'hm_size_{s}'
    if k in res:
        sizes.append(s)
        size_vals.append(res[k])
if sizes:
    plt.figure()
    plt.plot(sizes, size_vals, marker='o')
    plt.xlabel('heatmap_size')
    plt.ylabel('val_loss')
    plt.title('Ablation: Heatmap Size')
    plt.savefig('results/visualizations/ablation_size.png', bbox_inches='tight')
    plt.close()

sigmas = []
sigma_vals = []
for sg in [1.0,2.0,3.0,4.0]:
    k = f'sigma_{sg}'
    if k in res:
        sigmas.append(sg)
        sigma_vals.append(res[k])
if sigmas:
    plt.figure()
    plt.plot(sigmas, sigma_vals, marker='o')
    plt.xlabel('sigma')
    plt.ylabel('val_loss')
    plt.title('Ablation: Sigma')
    plt.savefig('results/visualizations/ablation_sigma.png', bbox_inches='tight')
    plt.close()
