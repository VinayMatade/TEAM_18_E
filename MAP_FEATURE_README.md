# MAP% Overfitting Monitor Feature

## Overview
Added a **MAP% (Mean Accuracy Percentage)** tracking feature to monitor model overfitting during training, similar to how mAP is used in YOLO object detection models.

## What is MAP%?
MAP% represents the prediction accuracy as a percentage of the data's value range:
- **100%** = Perfect predictions
- **0%** = Predictions are off by the entire data range
- Formula: `MAP% = max(0, 100 * (1 - MAE / value_range))`

## Features Added

### 1. Per-Epoch MAP% Display
During training, each epoch now shows:
```
Epoch 05/100 | Train Loss: 0.123456 | Val Loss: 0.098765 | MAP%: 87.45%
```

### 2. MAP% Visualization
Each training run generates a dual-plot showing:
- **Left plot**: Training and validation loss curves
- **Right plot**: MAP% progression over epochs with best MAP% indicator

### 3. Overfitting Detection
Monitor the MAP% curve to detect overfitting:
- **Rising MAP%**: Model is improving
- **Plateauing MAP%**: Model has converged
- **Declining MAP%**: Model may be overfitting (memorizing training data)

## How to Use

### During Training
Watch the console output for MAP% values:
- If MAP% stops improving while validation loss increases → **overfitting**
- If MAP% keeps improving → model is learning well
- If MAP% is low (<50%) → model may need more training or better hyperparameters

### After Training
Check the generated plots in the `plots/` directory:
- Files named `*_metrics.png` contain both loss and MAP% curves
- Look for the red dashed line showing the best MAP% achieved

## Example Interpretation

```
Epoch 10 | Train Loss: 0.050 | Val Loss: 0.045 | MAP%: 92.3%
Epoch 20 | Train Loss: 0.030 | Val Loss: 0.040 | MAP%: 94.1%  ← Improving
Epoch 30 | Train Loss: 0.015 | Val Loss: 0.042 | MAP%: 93.8%  ← Slight decline
Epoch 40 | Train Loss: 0.008 | Val Loss: 0.048 | MAP%: 92.5%  ← Overfitting!
```

In this example, the model starts overfitting around epoch 30-40 as:
- Training loss keeps decreasing
- Validation loss starts increasing
- MAP% starts declining

## Benefits
1. **Early overfitting detection**: Catch overfitting before it becomes severe
2. **Intuitive metric**: Percentage is easier to interpret than raw loss values
3. **Visual feedback**: Plots make trends immediately obvious
4. **Comparable across datasets**: Unlike loss values, MAP% is normalized

## Technical Details
- MAP% is calculated on the **validation set** after each epoch
- Predictions are **inverse-transformed** to original scale for meaningful percentages
- Computation is robust with fallback handling for edge cases
- Minimal performance overhead (computed during existing validation pass)
