# EE641 Homework 1

**Name:** Yaoxuan Liu  
**Email:** yaoxuanl@usc.edu  

## Problem1

### Train

This will generate:
- `results/training_log.json`  
- `results/best_model.pth`  
- `results/visualizations/loss_curve.png`

### Evaluate

This will generate:
- `results/visualizations/detections_img_01.png … detections_img_10.png`  
- `results/visualizations/anchor_coverage_by_scale.png`  
- `results/visualizations/scale_vs_object_size.png`  
- `results/visualizations/size_distribution.png`  
- `results/visualizations/scale_hits.png`  
- `results/visualizations/metrics.json`

## Deliverables
- `training_log.json` with loss values  
- `best_model.pth` with trained weights  
- `visualizations/` containing:
  - 10 detection result images  
  - anchor coverage visualization  
  - scale vs object size analysis  
  - training loss curve  
  - additional auxiliary plots (size distribution, scale hits)  

## Problem2 

This project compares two approaches for keypoint detection:
- Heatmap-based regression
- Direct regression

### run_ablation.py
Runs ablation experiments on heatmap resolution, Gaussian sigma, and skip connections.
Outputs results and plots in `results/`.

### makeresult.py
Generates evaluation results after training, including:
- PCK curves
- Predicted heatmaps at different epochs
- Sample predictions from both methods
- Failure case analysis

## Deliverables
- `results/training_log.json` – training curves for both methods  
- `results/heatmap_model.pth` – trained heatmap model  
- `results/regression_model.pth` – trained regression model  
- `results/visualizations/` – contains:
  - PCK curves comparing both methods  
  - Predicted heatmaps at different training stages  
  - Sample predictions from both methods on test images  
  - Failure case visualizations  
- `results/baseline_results` – ablation study results  


