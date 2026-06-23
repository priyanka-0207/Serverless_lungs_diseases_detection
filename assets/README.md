# Screenshots

Captured from the live deployment.

## Example predictions (one per class)
- **normal_prediction.png** — Normal, 98.19% confidence, with Grad-CAM heatmap
- **pneumonia_prediction.png** — Pneumonia, 99.93% confidence, with Grad-CAM heatmap
- **ild_prediction.png** — ILD, 99.53% confidence, with Grad-CAM heatmap

## System
- **dashboard.png** — full prediction result view (hero image)
- **xray_analysis.png** — original X-ray and Grad-CAM heatmap side by side
- **app_home.png** — landing page and AWS pipeline overview
- **app_upload.png** — upload / inference-in-progress state
- **cloudwatch.png** — CloudWatch dashboard: invocations, duration, errors

Note: prediction percentages are per-image confidence scores, not model accuracy.
Overall validation accuracy is 72% (macro F1 0.72).

## Still to add (optional)
- **confusion_matrix.png** — from the notebook's validation run; referenced by docs/RESULTS.md
