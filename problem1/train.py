import json
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from loss import DetectionLoss
from utils import generate_anchors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def train_epoch(model, dataloader, criterion, optimizer, device, anchors):
    model.train(); s=0.0; c=0
    for imgs, targets in dataloader:
        imgs=imgs.to(device); optimizer.zero_grad(); preds=model(imgs)
        loss,_=criterion(preds,targets,anchors); loss.backward(); optimizer.step()
        s+=float(loss.item()); c+=1
    return s/max(1,c)

def validate(model, dataloader, criterion, device, anchors):
    model.eval(); s=0.0; c=0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs=imgs.to(device); preds=model(imgs)
            loss,_=criterion(preds,targets,anchors); s+=float(loss.item()); c+=1
    return s/max(1,c)

def main():
    batch_size=16; learning_rate=0.001; num_epochs=50
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root=Path("/content/datasets/detection")
    images_train=data_root/"train"; ann_train=data_root/"train_annotations.json"
    images_val=data_root/"val";   ann_val=data_root/"val_annotations.json"
    train_set=ShapeDetectionDataset(str(images_train),str(ann_train))
    val_set=ShapeDetectionDataset(str(images_val),str(ann_val))
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,
                            collate_fn=lambda b:(torch.stack([x[0] for x in b],0),[x[1] for x in b]),num_workers=0)
    val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False,
                          collate_fn=lambda b:(torch.stack([x[0] for x in b],0),[x[1] for x in b]),num_workers=0)
    model=MultiScaleDetector(num_classes=3,num_anchors=3).to(device)
    criterion=DetectionLoss(num_classes=3)
    optimizer=optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
    dummy=torch.zeros(1,3,224,224).to(device)
    with torch.no_grad(): outs=model(dummy)
    feat_sizes=[(o.shape[2],o.shape[3]) for o in outs]
    anchor_scales=[[0.06,0.12,0.2],[0.25,0.35,0.45],[0.5,0.65,0.8]]
    anchors=generate_anchors(feat_sizes,anchor_scales,image_size=224)
    results_dir=Path("results"); results_dir.mkdir(parents=True,exist_ok=True)
    vis_dir=results_dir/"visualizations"; vis_dir.mkdir(parents=True,exist_ok=True)
    log_path=results_dir/"training_log.json"; best_model_path=results_dir/"best_model.pth"
    history={"train_loss":[],"val_loss":[]}; best_val=float("inf")
    for epoch in range(1,num_epochs+1):
        tr=train_epoch(model,train_loader,criterion,optimizer,device,anchors)
        va=validate(model,val_loader,criterion,device,anchors)
        history["train_loss"].append(tr); history["val_loss"].append(va)
        with open(log_path,"w",encoding="utf-8") as f: json.dump(history,f,indent=2)
        plt.figure()
        plt.plot(range(1,len(history["train_loss"])+1),history["train_loss"],label="train_loss")
        plt.plot(range(1,len(history["val_loss"])+1),history["val_loss"],label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training / Validation Loss")
        plt.legend(); plt.tight_layout(); plt.savefig(vis_dir/"loss_curve.png"); plt.close()
        if va<best_val: best_val=va; torch.save(model.state_dict(),best_model_path)
        print(f"Epoch {epoch:03d}/{num_epochs} - train {tr:.4f} - val {va:.4f} - best {best_val:.4f}")

if __name__=="__main__":
    main()
