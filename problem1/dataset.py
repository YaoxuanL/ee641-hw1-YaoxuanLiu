
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class ShapeDetectionDataset(Dataset):
    def __init__(self, image_dir: str, annotation_file: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        id_to_file = {}
        if isinstance(data.get("images"), list):
            for img in data["images"]:
                fname = img.get("file_name", img.get("filename"))
                if fname is None:
                    continue
                id_to_file[int(img["id"])] = fname

        per_img = {}
        if isinstance(data.get("annotations"), list):
            for ann in data["annotations"]:
                img_id = int(ann["image_id"])
                bbox = ann.get("bbox", None)
                cat = ann.get("category_id", 1)
                if bbox is None:
                    continue
                if img_id not in per_img:
                    per_img[img_id] = {"boxes": [], "labels": []}
                per_img[img_id]["boxes"].append([float(bbox[0]), float(bbox[1]), float(bbox[0]+bbox[2]), float(bbox[1]+bbox[3])])
                per_img[img_id]["labels"].append(int(cat))

        self._items = []
        for img_id, fname in id_to_file.items():
            p = self.image_dir / fname
            if not p.exists():
                p2 = self.image_dir / Path(fname).name
                if p2.exists():
                    p = p2
            entry = per_img.get(img_id, {"boxes": [], "labels": []})
            boxes = np.array(entry["boxes"], dtype=np.float32) if entry["boxes"] else np.zeros((0,4), dtype=np.float32)
            labels = np.array(entry["labels"], dtype=np.int64) if entry["labels"] else np.zeros((0,), dtype=np.int64)
            self._items.append((p, boxes, labels))

        if not self._items and isinstance(data, list):
            for it in data:
                p = self.image_dir / it["file_name"]
                bbox_list = it.get("bboxes", [])
                label_list = it.get("labels", [])
                boxes = np.array([[b[0],b[1],b[0]+b[2],b[1]+b[3]] for b in bbox_list], dtype=np.float32)
                labels = np.array(label_list, dtype=np.int64)
                self._items.append((p, boxes, labels))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx: int):
        img_path, boxes, labels = self._items[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img_tensor = self.transform(img)
        else:
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[2] == 4:
                arr = arr[..., :3]
            arr = arr / 255.0
            arr = np.transpose(arr, (2,0,1))
            img_tensor = torch.from_numpy(arr)
        targets = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "img_path": str(img_path)
        }
        return img_tensor, targets
