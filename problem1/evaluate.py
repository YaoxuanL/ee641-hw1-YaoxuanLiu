import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.ops import nms
from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from utils import generate_anchors, compute_iou

def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    flat=[]
    for i,pred in enumerate(predictions):
        if pred["boxes"].numel()==0: continue
        scores=pred["scores"].float().detach().cpu(); boxes=pred["boxes"].float().detach().cpu()
        for j in range(boxes.shape[0]): flat.append((i,float(scores[j].item()),boxes[j]))
    flat.sort(key=lambda x:-x[1])
    gts=[]
    for gt in ground_truths:
        b=gt["boxes"].float().detach().cpu()
        gts.append({"boxes":b,"used":torch.zeros((b.shape[0],),dtype=torch.bool)})
    T=int(sum([g["boxes"].shape[0] for g in gts]))
    if T==0: return 0.0
    tp_fp=[]
    for img_idx,_,box in flat:
        gt=gts[img_idx]["boxes"]
        if gt.numel()==0: tp_fp.append(0); continue
        x1=torch.max(box[0],gt[:,0]); y1=torch.max(box[1],gt[:,1])
        x2=torch.min(box[2],gt[:,2]); y2=torch.min(box[3],gt[:,3])
        inter=(x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)
        ap=(box[2]-box[0]).clamp(min=0)*(box[3]-box[1]).clamp(min=0)
        ag=(gt[:,2]-gt[:,0]).clamp(min=0)*(gt[:,3]-gt[:,1]).clamp(min=0)
        iou=torch.where(ap+ag-inter>0, inter/(ap+ag-inter), torch.zeros_like(ap))
        m,j=torch.max(iou,dim=0)
        if m.item()>=iou_threshold and not gts[img_idx]["used"][j]:
            tp_fp.append(1); gts[img_idx]["used"][j]=True
        else:
            tp_fp.append(0)
    tp=0; precisions=[]; recalls=[]
    for k,v in enumerate(tp_fp,1):
        tp+=v; precisions.append(tp/k); recalls.append(tp/T)
    ap=0.0
    for t in [i/10 for i in range(11)]:
        ap+=max([p for r,p in zip(recalls,precisions) if r>=t], default=0)/11.0
    return float(ap)

def visualize_detections(image,predictions,ground_truths,save_path):
    img=Image.open(image).convert("RGB")
    d=ImageDraw.Draw(img)
    if ground_truths["boxes"].numel()>0:
        for b in ground_truths["boxes"].tolist(): d.rectangle(b, outline=(0,255,0), width=2)
    if predictions["boxes"].numel()>0:
        for b in predictions["boxes"].tolist(): d.rectangle(b, outline=(255,0,0), width=2)
    Path(save_path).parent.mkdir(parents=True,exist_ok=True)
    img.save(save_path)

def analyze_scale_performance(model,dataloader,anchors):
    device=next(model.parameters()).device
    vis=Path("results/visualizations"); vis.mkdir(parents=True,exist_ok=True)
    all_areas=[]
    for _,tgts in dataloader:
        for t in tgts:
            if t["boxes"].numel()==0: continue
            for b in t["boxes"]:
                w=(b[2]-b[0]).item(); h=(b[3]-b[1]).item()
                all_areas.append(max(1.0,w*h))
    if len(all_areas)==0: all_areas=[1.0]
    q1,q2=np.quantile(all_areas,[0.33,0.66])
    def tag(b):
        a=max(1.0, float((b[2]-b[0])*(b[3]-b[1])))
        return "small" if a<=q1 else ("medium" if a<=q2 else "large")
    stats={"small":{"S1":0,"S2":0,"S3":0},"medium":{"S1":0,"S2":0,"S3":0},"large":{"S1":0,"S2":0,"S3":0}}
    hist={"small":0,"medium":0,"large":0}
    model.eval()
    with torch.no_grad():
        for imgs,tgts in dataloader:
            imgs=imgs.to(device); outs=model(imgs); anc=[a.to(device) for a in anchors]
            for b in range(imgs.shape[0]):
                for g in tgts[b]["boxes"]: hist[tag(g)]+=1
                for si in range(3):
                    B,C,H,W=outs[si].shape; A=anc[si].shape[0]//(H*W)
                    p=outs[si][b].permute(1,2,0).contiguous().view(H*W*A,5+getattr(model,"num_classes",3))
                    a=anc[si]
                    ax=(a[:,0]+a[:,2])*0.5; ay=(a[:,1]+a[:,3])*0.5
                    aw=(a[:,2]-a[:,0]).clamp(min=1e-6); ah=(a[:,3]-a[:,1]).clamp(min=1e-6)
                    tx,ty,tw,th=p[:,0],p[:,1],p[:,2],p[:,3]
                    obj=1/(1+torch.exp(-p[:,4]))
                    cls_prob=torch.softmax(p[:,5:],1).max(1)[0] if p[:,5:].numel()>0 else obj
                    cx=tx*aw+ax; cy=ty*ah+ay; w=aw*torch.exp(tw); h=ah*torch.exp(th)
                    x1=(cx-w/2).clamp(0,223); y1=(cy-h/2).clamp(0,223); x2=(cx+w/2).clamp(0,223); y2=(cy+h/2).clamp(0,223)
                    boxes=torch.stack([x1,y1,x2,y2],1); scores=obj*cls_prob
                    keep=scores>=0.6
                    boxes=boxes[keep]; scores=scores[keep]
                    if scores.numel()>400:
                        topk=torch.topk(scores,400).indices
                        boxes=boxes[topk]; scores=scores[topk]
                    if boxes.numel()==0: continue
                    keep_idx=nms(boxes, scores, 0.5)
                    boxes=boxes[keep_idx]
                    gtb=tgts[b]["boxes"].float().to(boxes.device)
                    if gtb.numel()==0 or boxes.numel()==0: continue
                    x1=torch.max(boxes[:,None,0],gtb[None,:,0]); y1=torch.max(boxes[:,None,1],gtb[None,:,1])
                    x2=torch.min(boxes[:,None,2],gtb[None,:,2]); y2=torch.min(boxes[:,None,3],gtb[None,:,3])
                    inter=(x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)
                    ap=(boxes[:,2]-boxes[:,0]).clamp(min=0)*(boxes[:,3]-boxes[:,1]).clamp(min=0)
                    ag=(gtb[:,2]-gtb[:,0]).clamp(min=0)*(gtb[:,3]-gtb[:,1]).clamp(min=0)
                    iou=inter/(ap[:,None]+ag[None,:]-inter+1e-9)
                    m,j=iou.max(1)
                    for k in range(m.shape[0]):
                        if m[k].item()>=0.5:
                            tname=tag(gtb[j[k]])
                            if   si==0: stats[tname]["S1"]+=1
                            elif si==1: stats[tname]["S2"]+=1
                            else:        stats[tname]["S3"]+=1
    plt.figure()
    plt.bar(["small","medium","large"], [hist["small"],hist["medium"],hist["large"]])
    Path(vis).mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(vis/"size_distribution.png"); plt.close()
    plt.figure()
    X=np.arange(3); W=0.25
    plt.bar(X-W,[stats[b]["S1"] for b in ["small","medium","large"]],width=W,label="S1")
    plt.bar(X,   [stats[b]["S2"] for b in ["small","medium","large"]],width=W,label="S2")
    plt.bar(X+W, [stats[b]["S3"] for b in ["small","medium","large"]],width=W,label="S3")
    plt.xticks(X,["small","medium","large"])
    plt.legend(); plt.tight_layout(); plt.savefig(vis/"scale_vs_object_size.png"); plt.close()

def generate_all_visualizations():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root=Path("/content/datasets/detection")
    val_set=ShapeDetectionDataset(str(data_root/"val"), str(data_root/"val_annotations.json"))
    val_loader=DataLoader(val_set,batch_size=2,shuffle=False,
                          collate_fn=lambda b:(torch.stack([x[0] for x in b],0),[x[1] for x in b]),num_workers=0)
    Path("results/visualizations").mkdir(parents=True, exist_ok=True)
    model=MultiScaleDetector(num_classes=3,num_anchors=3).to(device)
    model.load_state_dict(torch.load("results/best_model.pth",map_location=device))
    model.eval()
    dummy=torch.zeros(1,3,224,224).to(device)
    with torch.no_grad(): outs=model(dummy)
    feat_sizes=[(o.shape[2],o.shape[3]) for o in outs]
    anchors=generate_anchors(feat_sizes,[[16,24,32],[48,64,96],[96,128,192]],224)
    analyze_scale_performance(model,val_loader,anchors)
    saved=0; vis=Path("results/visualizations")
    with torch.no_grad():
        for imgs,tgts in val_loader:
            imgs=imgs.to(device); outs=model(imgs)
            merged=[]
            for si in range(3):
                B,C,H,W=outs[si].shape; A=anchors[si].shape[0]//(H*W)
                merged.append(outs[si].permute(0,2,3,1).contiguous().view(B,H*W*A,5+getattr(model,"num_classes",3)))
            preds_all=torch.cat(merged,1)
            for b in range(imgs.shape[0]):
                img_np=(imgs[b].detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).astype("uint8")
                im=Image.fromarray(img_np); dr=ImageDraw.Draw(im)
                for bb in tgts[b]["boxes"].tolist(): dr.rectangle(bb, outline=(0,255,0), width=2)
                im.save(vis/f"detections_img_{saved+1:02d}.png"); saved+=1
                if saved>=10: break
            if saved>=10: break
    counts={"S1":0,"S2":0,"S3":0}
    with torch.no_grad():
        for _,tgts in val_loader:
            for t in tgts:
                if t["boxes"].numel()==0: continue
                acats=[]; rng=[]; s=0
                for a in anchors:
                    acats.append(a); e=s+a.shape[0]; rng.append((s,e)); s=e
                acats=torch.cat(acats,0)
                ious=compute_iou(t["boxes"].float(), acats.float())
                _,best=ious.max(dim=1)
                for idx in best.tolist():
                    if rng[0][0]<=idx<rng[0][1]: counts["S1"]+=1
                    elif rng[1][0]<=idx<rng[1][1]: counts["S2"]+=1
                    else: counts["S3"]+=1
    plt.figure()
    plt.bar(["S1","S2","S3"], [counts["S1"],counts["S2"],counts["S3"]])
    plt.tight_layout(); plt.savefig(vis/"anchor_coverage_by_scale.png"); plt.close()
    with open("results/training_log.json","r",encoding="utf-8") as f:
        hist=json.load(f)
    plt.figure()
    if isinstance(hist,dict) and "train_loss" in hist:
        plt.plot(hist["train_loss"],label="train"); 
        if "val_loss" in hist: plt.plot(hist["val_loss"],label="val")
    elif isinstance(hist,dict) and "heatmap" in hist:
        plt.plot(hist["heatmap"].get("train",[]),label="train(hm)")
        plt.plot(hist["heatmap"].get("val",[]),label="val(hm)")
    plt.legend(); plt.tight_layout(); plt.savefig(vis/"loss_curve.png"); plt.close()

if __name__=="__main__":
    generate_all_visualizations()

