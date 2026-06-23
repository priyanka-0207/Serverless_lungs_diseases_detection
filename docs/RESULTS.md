# Results

Metrics are computed on a held-out validation set. The train/validation split is performed **before** any oversampling, so minority-class oversampling is applied to the training set only and does not leak into validation.

## Validation metrics

| Metric | Value |
|---|---|
| Accuracy | 72% |
| Macro F1 | 0.72 |

### Per-class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | _[paste from classification_report]_ | _[..]_ | _[..]_ | _[..]_ |
| Pneumonia | _[paste from classification_report]_ | _[..]_ | _[..]_ | _[..]_ |
| ILD | _[paste from classification_report]_ | _[..]_ | _[..]_ | _[..]_ |

> Paste the per-class rows from the notebook's `classification_report` output. The headline accuracy and macro F1 are already filled in.

## Confusion matrix

Drop the image at `assets/confusion_matrix.png` and it will render here:

![Confusion matrix](../assets/confusion_matrix.png)

## Notes

- 72% accuracy / 0.72 macro F1 on a 3-class problem (Normal, Pneumonia, ILD) over a 5,307-image NIH ChestX-ray14 subset.
- Pneumonia had only 307 real images and was oversampled in the training set to balance classes. Validation contains only real, non-duplicated images.
- These are validation-set metrics on a small dataset, not a measure of real-world clinical performance. This is a research and engineering demo, not a medical device.
