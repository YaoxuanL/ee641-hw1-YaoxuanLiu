import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import os

class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap', 
                 heatmap_size=64, sigma=2.0):
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        self.items = []
        if isinstance(data, dict) and 'images' in data and isinstance(data['images'], list):
            for it in data['images']:
                fn = it.get('file_name')
                ks = it.get('keypoints')
                if fn is None or not isinstance(ks, list) or len(ks) != 5:
                    continue
                if not all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in ks):
                    continue
                kps = np.array([[float(p[0]), float(p[1])] for p in ks], dtype=np.float32)
                p = os.path.join(self.image_dir, fn)
                if os.path.exists(p):
                    self.items.append((p, kps))

    def __len__(self):
        return len(self.items)

    def generate_heatmap(self, keypoints, height, width):
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        xx = xx.float()
        yy = yy.float()
        hms = []
        for i in range(keypoints.shape[0]):
            x, y = keypoints[i]
            x = torch.clamp(torch.tensor(x, dtype=torch.float32), 0, width - 1)
            y = torch.clamp(torch.tensor(y, dtype=torch.float32), 0, height - 1)
            g = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * (self.sigma**2)))
            hms.append(g)
        return torch.stack(hms, dim=0)

    def __getitem__(self, idx):
        path, kps = self.items[idx]
        img = Image.open(path).convert('L').resize((128, 128))
        img = torch.from_numpy(np.array(img, dtype=np.uint8)).float().unsqueeze(0) / 255.0
        if self.output_type == 'heatmap':
            sx = self.heatmap_size / 128.0
            sy = self.heatmap_size / 128.0
            kps_scaled = np.stack([kps[:, 0] * sx, kps[:, 1] * sy], axis=1)
            tgt = self.generate_heatmap(torch.from_numpy(kps_scaled).float(), self.heatmap_size, self.heatmap_size)
            return img, tgt
        xs = kps[:, 0] / 128.0
        ys = kps[:, 1] / 128.0
        tgt = np.concatenate([xs, ys], axis=0).astype(np.float32)
        return img, torch.from_numpy(tgt)
