import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import matplotlib.pyplot as plt
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet

def train_heatmap_model(model, train_loader, val_loader, num_epochs=30):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = next(model.parameters()).device
    best = float('inf')
    log = {'train': [], 'val': []}
    for _ in range(num_epochs):
        model.train()
        tr, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            o = model(x)
            l = criterion(o, y)
            l.backward()
            optimizer.step()
            tr += l.item() * x.size(0)
            n += x.size(0)
        tr /= max(n, 1)
        model.eval()
        vl, m = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                o = model(x)
                l = criterion(o, y)
                vl += l.item() * x.size(0)
                m += x.size(0)
        vl /= max(m, 1)
        log['train'].append(tr)
        log['val'].append(vl)
        e = len(log['val'])
        if e in {10, 20, 30}:
            torch.save(model.state_dict(), f'results/heatmap_epoch_{e}.pth')
        if vl < best:
            best = vl
            torch.save(model.state_dict(), 'results/heatmap_model.pth')
    return log

def train_regression_model(model, train_loader, val_loader, num_epochs=30):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = next(model.parameters()).device
    best = float('inf')
    log = {'train': [], 'val': []}
    for _ in range(num_epochs):
        model.train()
        tr, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            o = model(x)
            l = criterion(o, y)
            l.backward()
            optimizer.step()
            tr += l.item() * x.size(0)
            n += x.size(0)
        tr /= max(n, 1)
        model.eval()
        vl, m = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                o = model(x)
                l = criterion(o, y)
                vl += l.item() * x.size(0)
                m += x.size(0)
        vl /= max(m, 1)
        log['train'].append(tr)
        log['val'].append(vl)
        e = len(log['val'])
        if e in {10, 20, 30}:
            torch.save(model.state_dict(), f'results/regression_epoch_{e}.pth')
        if vl < best:
            best = vl
            torch.save(model.state_dict(), 'results/regression_model.pth')
    return log

def main():
    os.makedirs('results/visualizations', exist_ok=True)
    train_img = '/content/datasets/keypoints/train'
    val_img = '/content/datasets/keypoints/val'
    train_ann = '/content/datasets/keypoints/train_annotations.json'
    val_ann = '/content/datasets/keypoints/val_annotations.json'
    train_ds_h = KeypointDataset(train_img, train_ann, output_type='heatmap', heatmap_size=64, sigma=2.0)
    val_ds_h = KeypointDataset(val_img, val_ann, output_type='heatmap', heatmap_size=64, sigma=2.0)
    train_ds_r = KeypointDataset(train_img, train_ann, output_type='regression', heatmap_size=64, sigma=2.0)
    val_ds_r = KeypointDataset(val_img, val_ann, output_type='regression', heatmap_size=64, sigma=2.0)
    train_l_h = DataLoader(train_ds_h, batch_size=32, shuffle=True, num_workers=0)
    val_l_h = DataLoader(val_ds_h, batch_size=32, shuffle=False, num_workers=0)
    train_l_r = DataLoader(train_ds_r, batch_size=32, shuffle=True, num_workers=0)
    val_l_r = DataLoader(val_ds_r, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hm = HeatmapNet().to(device)
    rg = RegressionNet().to(device)
    log_h = train_heatmap_model(hm, train_l_h, val_l_h, num_epochs=30)
    log_r = train_regression_model(rg, train_l_r, val_l_r, num_epochs=30)
    with open('results/training_log.json', 'w') as f:
        json.dump({'heatmap': log_h, 'regression': log_r}, f, indent=2)
    epochs = list(range(1, len(log_h['train']) + 1))
    plt.figure()
    plt.plot(epochs, log_h['train'], label='heatmap-train')
    plt.plot(epochs, log_h['val'], label='heatmap-val')
    plt.plot(epochs, log_r['train'], label='regression-train')
    plt.plot(epochs, log_r['val'], label='regression-val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('results/visualizations/training_curves.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
