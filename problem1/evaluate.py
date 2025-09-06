from pathlib import Path
import json, math
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from utils import generate_anchors, compute_iou
import numpy as np

def compute_ap(predictions, ground_truths, iou_threshold=0.5, num_classes=3):
    aps=[]
    for c in range(1,num_classes+1):
        P=[]; all_preds=[]
        for img_idx,pred in enumerate(predictions):
            mask=(pred["labels"]==c); idxs=torch.nonzero(mask).flatten()
            for i in idxs.tolist():
                all_preds.append((img_idx,float(pred["scores"][i].item()),pred["boxes"][i]))
        all_preds.sort(key=lambda x:-x[1])
        gt_used=[torch.zeros((gt["boxes"].shape[0],),dtype=torch.bool) for gt in ground_truths]
        T=int(sum([(gt["labels"]==c).sum().item() for gt in ground_truths]))
        for img_idx,score,box in all_preds:
            gt=ground_truths[img_idx]; gt_mask=(gt["labels"]==c); boxes_gt=gt["boxes"][gt_mask]
            if boxes_gt.shape[0]==0: P.append(0); continue
            ious=compute_iou(box.unsqueeze(0),boxes_gt)[0]; max_iou,j=torch.max(ious,0)
            if max_iou.item()>=iou_threshold and not gt_used[img_idx][torch.nonzero(gt_mask).flatten()[j]]:
                P.append(1); gt_used[img_idx][torch.nonzero(gt_mask).flatten()[j]]=True
            else: P.append(0)
        if T==0: continue
        tp=0; precisions=[]; recalls=[]
        for k,p in enumerate(P,1):
            tp+=p; precisions.append(tp/k); recalls.append(tp/T)
        ap=0.0
        for t in [i/10 for i in range(11)]:
            prec_at_r=0.0
            for r,p in zip(recalls,precisions):
                if r>=t and p>prec_at_r: prec_at_r=p
            ap+=prec_at_r/11.0
        aps.append(ap)
    return float(sum(aps)/max(1,len(aps))) if aps else 0.0

def visualize_detections(image, predictions, ground_truths, save_path):
    img=Image.open(image).convert("RGB"); draw=ImageDraw.Draw(img)
    for b in ground_truths["boxes"].tolist(): draw.rectangle(b,outline=(0,255,0),width=2)
    for b in predictions["boxes"].tolist(): draw.rectangle(b,outline=(255,0,0),width=2)
    img.save(save_path)

def analyze_scale_performance(model, dataloader, anchors, save_dir):
    save_dir=Path(save_dir); save_dir.mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device
    size_bins={"small":0,"medium":0,"large":0}; hits={"s":0,"m":0,"l":0}
    with torch.no_grad():
        for imgs,targets in dataloader:
            imgs=imgs.to(device); outs=model(imgs)
            for b in range(imgs.shape[0]):
                for gt in targets[b]["boxes"]:
                    w=(gt[2]-gt[0]).item(); h=(gt[3]-gt[1]).item(); area=math.sqrt(max(1.0,w*h))
                    if area<32: size_bins["small"]+=1
                    elif area<96: size_bins["medium"]+=1
                    else: size_bins["large"]+=1
                for si in range(3):
                    B,ch,H,W=outs[si].shape; A=anchors[si].shape[0]//(H*W)
                    pred=outs[si][b].permute(1,2,0).contiguous().view(-1,A,5+model.num_classes)
                    obj=pred[...,4].reshape(-1)
                    if obj.numel()>0 and obj.max().item()>0:
                        if si==0: hits["s"]+=1
                        elif si==1: hits["m"]+=1
                        else: hits["l"]+=1
    plt.figure(); plt.bar(list(size_bins.keys()),list(size_bins.values()))
    plt.title("GT object size distribution"); plt.savefig(save_dir/"size_distribution.png"); plt.close()
    plt.figure(); plt.bar(list(hits.keys()),list(hits.values()))
    plt.title("Scale activations (proxy)"); plt.savefig(save_dir/"scale_hits.png"); plt.close()

if __name__=="__main__":
    import torch
    from torch.utils.data import DataLoader
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root=Path("/content/datasets/detection")
    images_val=data_root/"val"; ann_val=data_root/"val_annotations.json"
    val_set=ShapeDetectionDataset(str(images_val),str(ann_val))
    val_loader=DataLoader(val_set,batch_size=8,shuffle=False,
                          collate_fn=lambda b:(torch.stack([x[0] for x in b],0),[x[1] for x in b]),num_workers=0)
    model=MultiScaleDetector(num_classes=3,num_anchors=3).to(device)
    model.load_state_dict(torch.load("results/best_model.pth",map_location=device)); model.eval()
    dummy=torch.zeros(1,3,224,224).to(device)
    with torch.no_grad(): outs=model(dummy)
    feat_sizes=[(o.shape[2],o.shape[3]) for o in outs]
    anchor_scales=[[0.06,0.12,0.2],[0.25,0.35,0.45],[0.5,0.65,0.8]]
    anchors=generate_anchors(feat_sizes,anchor_scales,image_size=224)
    predictions=[]; ground_truths=[]
    vis_dir=Path("results/visualizations"); vis_dir.mkdir(parents=True,exist_ok=True)
    saved=0
    with torch.no_grad():
        for imgs,targets in val_loader:
            imgs=imgs.to(device); outs=model(imgs)
            anc_all=torch.cat([a.to(device) for a in anchors],0)
            merged=[]
            for si in range(3):
                B,ch,H,W=outs[si].shape; A=anchors[si].shape[0]//(H*W)
                merged.append(outs[si].permute(0,2,3,1).contiguous().view(B,H*W*A,5+model.num_classes))
            preds_all=torch.cat(merged,1)
            for b in range(imgs.shape[0]):
                ax=(anc_all[:,0]+anc_all[:,2])*0.5; ay=(anc_all[:,1]+anc_all[:,3])*0.5
                aw=(anc_all[:,2]-anc_all[:,0]).clamp(min=1e-6); ah=(anc_all[:,3]-anc_all[:,1]).clamp(min=1e-6)
                p=preds_all[b]; tx,ty,tw,th=p[:,0],p[:,1],p[:,2],p[:,3]
                obj=1.0/(1.0+torch.exp(-p[:,4])); cls_logits=p[:,5:]
                if cls_logits.numel()==0:
                    cls_prob=obj.new_ones(obj.shape); cls_id=torch.zeros_like(obj,dtype=torch.long)
                else:
                    probs=torch.softmax(cls_logits,1); cls_prob,cls_id=probs.max(1)
                cx=tx*aw+ax; cy=ty*ah+ay; w=aw*torch.exp(tw); h=ah*torch.exp(th)
                x1=(cx-w/2).clamp(0,223); y1=(cy-h/2).clamp(0,223); x2=(cx+w/2).clamp(0,223); y2=(cy+h/2).clamp(0,223)
                boxes=torch.stack([x1,y1,x2,y2],1); scores=(obj*cls_prob)
                keep=scores>=0.3; boxes=boxes[keep]; scores=scores[keep]; cls_id=cls_id[keep]
                order=torch.argsort(scores,descending=True); keep_idx=[]
                while order.numel()>0:
                    i=int(order[0].item()); keep_idx.append(i)
                    if order.numel()==1: break
                    ious=compute_iou(boxes[i].unsqueeze(0),boxes[order[1:]])[0]
                    order=order[1:][ious<0.5]
                boxes=boxes[keep_idx].cpu(); scores=scores[keep_idx].cpu(); cls_id=cls_id[keep_idx].cpu()
                predictions.append({"boxes":boxes,"scores":scores,"labels":cls_id})
                ground_truths.append({"boxes":targets[b]["boxes"],"labels":targets[b]["labels"]})
                if saved<10:
                    img_np=(imgs[b].detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).astype('uint8')
                    img_pil=Image.fromarray(img_np); draw=ImageDraw.Draw(img_pil)
                    for bb in ground_truths[-1]["boxes"].tolist(): draw.rectangle(bb,outline=(0,255,0),width=2)
                    for bb in boxes.tolist(): draw.rectangle(bb,outline=(255,0,0),width=2)
                    img_pil.save(vis_dir/f"detections_img_{saved+1:02d}.png"); saved+=1
    mAP50=compute_ap(predictions,ground_truths,0.5,3)
    with open(vis_dir/"metrics.json","w",encoding="utf-8") as f: json.dump({"mAP@0.5":mAP50},f,indent=2)
    anchor_counts={"S1":0,"S2":0,"S3":0}
    size_bins={"small":0,"medium":0,"large":0}
    scale_vs_size={"small":{"S1":0,"S2":0,"S3":0},
                   "medium":{"S1":0,"S2":0,"S3":0},
                   "large":{"S1":0,"S2":0,"S3":0}}
    all_areas=[]
    for gt in ground_truths:
        for b in gt["boxes"]:
            w=(b[2]-b[0]).item(); h=(b[3]-b[1]).item(); all_areas.append(max(1.0,w*h))
    if len(all_areas)==0: all_areas=[1.0]
    q1,q2=np.quantile(all_areas,[0.33,0.66])
    for gt in ground_truths:
        boxes_gt=gt["boxes"]
        if boxes_gt.numel()==0: continue
        anc_cat=[]; ranges=[]; start=0
        for a in anchors:
            anc_cat.append(a); end=start+a.shape[0]; ranges.append((start,end)); start=end
        anc_cat=torch.cat(anc_cat,0)
        ious=compute_iou(boxes_gt,anc_cat); _,best_idx=ious.max(1)
        for gi,idx in enumerate(best_idx.tolist()):
            sc="S1" if ranges[0][0]<=idx<ranges[0][1] else ("S2" if ranges[1][0]<=idx<ranges[1][1] else "S3")
            anchor_counts[sc]+=1
            b=boxes_gt[gi]; area=float((b[2]-b[0])*(b[3]-b[1]))
            tag="small" if area<=q1 else ("medium" if area<=q2 else "large")
            size_bins[tag]+=1; scale_vs_size[tag][sc]+=1
    plt.figure()
    xs=["S1(high-res)","S2(med-res)","S3(low-res)"]; ys=[anchor_counts["S1"],anchor_counts["S2"],anchor_counts["S3"]]
    plt.bar(xs,ys); plt.title("Anchor coverage per scale (best-IoU GT→Anchor)")
    plt.ylabel("#GT best-matched"); plt.tight_layout(); plt.savefig(vis_dir/"anchor_coverage_by_scale.png"); plt.close()
    plt.figure()
    buckets=["small","medium","large"]; X=np.arange(len(buckets)); W=0.25
    plt.bar(X-W,[scale_vs_size[b]["S1"] for b in buckets],width=W,label="S1")
    plt.bar(X,[scale_vs_size[b]["S2"] for b in buckets],width=W,label="S2")
    plt.bar(X+W,[scale_vs_size[b]["S3"] for b in buckets],width=W,label="S3")
    plt.xticks(X,buckets); plt.ylabel("#GT best-matched")
    plt.title("Which scale detects which object sizes (GT→best Anchor)")
    plt.legend(); plt.tight_layout(); plt.savefig(vis_dir/"scale_vs_object_size.png"); plt.close()
