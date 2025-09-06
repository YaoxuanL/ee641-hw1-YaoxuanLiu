import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class ShapeDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = []

        if isinstance(data, dict) and "images" in data and "annotations" in data:
            id2img = {}
            for im in data["images"]:
                id2img[im["id"]] = {
                    "file_name": im["file_name"],
                    "width": im.get("width", None),
                    "height": im.get("height", None),
                }
            name_to_idx = {"circle": 0, "square": 1, "triangle": 2}
            catid_to_label = {}
            if "categories" in data:
                for c in data["categories"]:
                    nm = str(c.get("name", "")).strip().lower()
                    catid_to_label[c["id"]] = name_to_idx.get(nm, int(c["id"]) % 3)
            img_to_boxes, img_to_labels = {}, {}
            for ann in data["annotations"]:
                img_id = ann["image_id"]
                x, y, w, h = ann["bbox"]
                x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
                lab = catid_to_label.get(ann["category_id"], int(ann["category_id"]) % 3)
                img_to_boxes.setdefault(img_id, []).append([x1, y1, x2, y2])
                img_to_labels.setdefault(img_id, []).append(int(lab))
            for img_id, meta in id2img.items():
                self.items.append({
                    "file_name": meta["file_name"],
                    "width": meta.get("width", None),
                    "height": meta.get("height", None),
                    "boxes": img_to_boxes.get(img_id, []),
                    "labels": img_to_labels.get(img_id, []),
                })
        else:
            imgs = data["images"] if isinstance(data, dict) and "images" in data else data
            for im in imgs:
                self.items.append({
                    "file_name": im["file_name"],
                    "width": im.get("width", None),
                    "height": im.get("height", None),
                    "boxes": im.get("boxes", []),
                    "labels": [int(l) for l in im.get("labels", [])],
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.image_dir, item["file_name"])
        img = Image.open(img_path).convert("RGB")
        ow, oh = img.size
        tw, th = 224, 224
        sx = tw / float(ow)
        sy = th / float(oh)
        img = img.resize((tw, th), resample=Image.BILINEAR)

        boxes_list = item.get("boxes", [])
        if boxes_list:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            boxes[:, 0] = (boxes[:, 0] * sx).clamp(0, tw - 1)
            boxes[:, 1] = (boxes[:, 1] * sy).clamp(0, th - 1)
            boxes[:, 2] = (boxes[:, 2] * sx).clamp(0, tw - 1)
            boxes[:, 3] = (boxes[:, 3] * sy).clamp(0, th - 1)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels_list = item.get("labels", [])
        labels = torch.tensor(labels_list, dtype=torch.long) if labels_list else torch.zeros((0,), dtype=torch.long)

        if self.transform is not None:
            img = self.transform(img)
        else:
            import numpy as np
            arr = np.array(img, dtype="uint8")
            img = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0

        targets = {"boxes": boxes, "labels": labels}
        return img, targets
