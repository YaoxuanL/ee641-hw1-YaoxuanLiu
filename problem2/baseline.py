def ablation_study(dataset, model_class):
    import torch, json, os
    from torch.utils.data import DataLoader
    from dataset import KeypointDataset
    from model import HeatmapNet
    from train import train_heatmap_model
    from types import MethodType
    import torch.nn.functional as F

    os.makedirs('results', exist_ok=True)
    res = {}
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(ds):
        dl_tr = DataLoader(ds['train'], batch_size=32, shuffle=True, num_workers=0)
        dl_va = DataLoader(ds['val'], batch_size=32, shuffle=False, num_workers=0)
        m = HeatmapNet().to(dev)
        target_h = ds['train'].heatmap_size
        orig_forward = m.forward
        def patched_forward(self, x, _orig=orig_forward, _h=target_h):
            y = _orig(x)
            if y.shape[-1] != _h:
                y = F.interpolate(y, size=(_h, _h), mode='bilinear', align_corners=False)
            return y
        m.forward = MethodType(patched_forward, m)
        log = train_heatmap_model(m, dl_tr, dl_va, num_epochs=10)
        return log.get('val', log.get('heatmap_val'))[-1]

    for s in [32,64,128]:
        dtr = KeypointDataset(dataset['train'][0], dataset['train'][1], output_type='heatmap', heatmap_size=s, sigma=2.0)
        dva = KeypointDataset(dataset['val'][0], dataset['val'][1], output_type='heatmap', heatmap_size=s, sigma=2.0)
        res[f'hm_size_{s}'] = run({'train':dtr,'val':dva})

    for sg in [1.0,2.0,3.0,4.0]:
        dtr = KeypointDataset(dataset['train'][0], dataset['train'][1], output_type='heatmap', heatmap_size=64, sigma=sg)
        dva = KeypointDataset(dataset['val'][0], dataset['val'][1], output_type='heatmap', heatmap_size=64, sigma=sg)
        res[f'sigma_{sg}'] = run({'train':dtr,'val':dva})

    dtr = KeypointDataset(dataset['train'][0], dataset['train'][1], output_type='heatmap', heatmap_size=64, sigma=2.0)
    dva = KeypointDataset(dataset['val'][0], dataset['val'][1], output_type='heatmap', heatmap_size=64, sigma=2.0)
    m_with = HeatmapNet().to(dev)

    def no_skip_forward(self, x):
        import torch
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        d4 = self.deconv4(e4)
        z3 = torch.zeros_like(e3)
        d3 = self.deconv3(torch.cat([d4, z3], dim=1))
        z2 = torch.zeros_like(e2)
        d2 = self.deconv2(torch.cat([d3, z2], dim=1))
        return self.final(d2)

    from types import MethodType
    m_without = HeatmapNet().to(dev)
    m_without.forward = MethodType(no_skip_forward, m_without)

    dl_tr = DataLoader(dtr, batch_size=32, shuffle=True, num_workers=0)
    dl_va = DataLoader(dva, batch_size=32, shuffle=False, num_workers=0)
    from train import train_heatmap_model as th
    log_with = th(m_with, dl_tr, dl_va, num_epochs=10)
    log_without = th(m_without, dl_tr, dl_va, num_epochs=10)
    res['skip_with'] = log_with.get('val', log_with.get('heatmap_val'))[-1]
    res['skip_without'] = log_without.get('val', log_without.get('heatmap_val'))[-1]

    with open('results/baseline_results.json','w') as f:
        json.dump(res, f)
    return res

def analyze_failure_cases(model, test_loader):
    import torch, json, os
    from evaluate import extract_keypoints_from_heatmaps
    os.makedirs('results', exist_ok=True)
    out = {'H_ok_R_bad':[], 'R_ok_H_bad':[], 'both_bad':[]}
    try:
        if not isinstance(model, dict) or 'heatmap' not in model or 'regression' not in model:
            with open('results/failure_cases.json','w') as f:
                json.dump(out, f)
            return out
        mh = model['heatmap']
        mr = model['regression']
        dev = next(mh.parameters()).device
        mh.eval(); mr.eval()
        thr = 0.05
        idx = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(dev)
                if isinstance(y, torch.Tensor):
                    y = y.to(dev)
                if isinstance(y, torch.Tensor) and y.dim() == 4:
                    yk = extract_keypoints_from_heatmaps(y)
                elif isinstance(y, torch.Tensor) and y.dim() == 2 and y.size(1) == 10:
                    yk = y.view(y.size(0), -1, 2) * 64.0
                elif isinstance(y, torch.Tensor) and y.dim() == 3 and y.size(2) == 2:
                    yk = y
                else:
                    yk = y
                ph = extract_keypoints_from_heatmaps(mh(x))
                pr = mr(x).view(x.size(0), -1, 2) * 64.0
                x_min, _ = yk[:,:,0].min(dim=1)
                y_min, _ = yk[:,:,1].min(dim=1)
                x_max, _ = yk[:,:,0].max(dim=1)
                y_max, _ = yk[:,:,1].max(dim=1)
                diag = torch.sqrt((x_max - x_min)**2 + (y_max - y_min)**2) + 1e-6
                dh = (ph - yk).norm(dim=2)
                dr = (pr - yk).norm(dim=2)
                h_ok = (dh <= diag.unsqueeze(1)*thr).float().mean(dim=1) >= 0.8
                r_ok = (dr <= diag.unsqueeze(1)*thr).float().mean(dim=1) >= 0.8
                for i in range(x.size(0)):
                    if h_ok[i] and not r_ok[i]:
                        out['H_ok_R_bad'].append(int(idx+i))
                    elif r_ok[i] and not h_ok[i]:
                        out['R_ok_H_bad'].append(int(idx+i))
                    elif not h_ok[i] and not r_ok[i]:
                        out['both_bad'].append(int(idx+i))
                idx += x.size(0)
    except Exception as e:
        out = {'error': str(e)}
    with open('results/failure_cases.json','w') as f:
        json.dump(out, f)
    return out
