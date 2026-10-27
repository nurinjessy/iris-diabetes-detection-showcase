# Explainable AI for Iris-Based Diabetes Detection

Final Year Project (Bachelor of Computer and Communication Systems Engineering, Universiti Putra Malaysia, 2026).

A non-invasive screening pipeline that classifies iris images as **diabetic** or **healthy**, and uses **LIME** to show *which regions of the iris* drove each prediction.

> \*\*Portfolio showcase only.\*\* The full source code is copyrighted and is not published here. This repository contains documentation, screenshots and a few short excerpts. All rights reserved. 

\---

## Why this project

Many iris-based diabetes classifiers report accuracy but give no insight into *why* an image was classified a certain way. In healthcare, that makes them hard to trust. This project builds a complete pipeline and makes its decisions inspectable.

## Pipeline

```
Raw iris image
   → CLAHE contrast enhancement
   → Pupil detection (adaptive threshold + contours, Hough fallback)
   → Iris segmentation (iris radius = 3.3 × pupil radius, tuned empirically)
   → Letterbox resize to 288 × 192
   → HOG (6,732 features) + multi-scale LBP (uniform, radii 1/2/3)
   → Feature selection: mutual information + SelectKBest (top 200)
   → SMOTE on the training set only
   → Random Forest (300 trees) + Platt-scaling calibration
   → LIME explanations mapped back onto the iris image
```

## Results

Evaluated on a held-out, stratified 30% test set (96 images).

|Metric|Value|
|-|-|
|Accuracy|84.38%|
|Precision|84.09%|
|Recall|82.22%|
|F1-score|83.15%|
|ROC-AUC|0.9015|

Test confusion matrix: 44 healthy and 37 diabetic images correct; 7 healthy flagged as diabetic; 8 diabetic missed.

## Explainability

LIME was applied to the Random Forest's predictions. Because feature selection keeps a *subset of the original* HOG/LBP features (instead of compressing them with PCA), every selected feature still maps to a location and gradient orientation on the iris. This lets the system draw:

* HOG orientation "starburst" overlays, colour-coded by gradient angle
* LIME heatmaps (warm = pushes towards diabetic, cool = pushes towards healthy)
* Bounding boxes around the most influential iris regions

<!-- Add your screenshots below. Example:
!\[LIME heatmap](docs/screenshots/lime\_heatmap.png)

## Interactive dashboard

I built a single-page dashboard, "Healthcare AI Tools for Early Diabetes Detection", to present the model's reasoning to a non-technical viewer. It is plain HTML, CSS and JavaScript in one file, with no backend.

**User flow**

1. **Intake:** upload an iris photo or pick one of two sample patient cases. Diagnosis labels are hidden until the analysis runs, like a real unlabelled scan.
2. **Processing:** an animated scanner steps through the pipeline: isolating the iris, CLAHE enhancement, HOG + LBP feature extraction, Random Forest classification, and LIME explanation.
3. **Results:** the dashboard shows

   * the AI prediction (Diabetic or Healthy) with an evidence meter showing the balance of supporting regions
   * a draggable before/after slider comparing the raw iris with its CLAHE-enhanced version
   * the LIME explanation map (warm = supports Diabetic, cool = supports Healthy)
   * a plain-language explanation of why the model reached its result, and a ranked list of the most influential iris regions by position and strength

**Design choices**

* Written for clinicians and the general public, not data scientists. LIME is explained as "re-running the prediction hundreds of times while hiding parts of the texture."
* Respects `prefers-reduced-motion`, and the upload area and slider are keyboard accessible.
* Clearly labelled as a research prototype.

**What the demo does and does not do**

This is an offline demonstration. The results for the two sample cases were pre-computed with the real pipeline above and embedded in the page. Uploaded images are matched to the samples with a simple average-hash comparison, and unrecognised images get a message that live inference needs a connected model server. The dashboard does not run the model in the browser.

<!-- Add a dashboard screenshot:
!\[Dashboard results view](docs/screenshots/dashboard.png)

## Design decisions I made

* **Mutual information + SelectKBest instead of PCA.** PCA would have merged features into abstract components, losing the local texture differences between eyes and breaking the link between a feature and an iris region, which the explainability layer depends on.
* **Leakage-aware ordering.** The train/test split happens first. Feature selection and SMOTE are fitted on the training set only.
* **Classical ML over deep learning.** With about 300 images, handcrafted features and a Random Forest are less prone to overfitting and easier to explain.
* **Fixed segmentation parameters.** Tuned on a representative sample from each class, then frozen for the whole batch run.

## Tech stack

Python · OpenCV · scikit-image · scikit-learn · imbalanced-learn (SMOTE) · LIME · NumPy / SciPy · Matplotlib · HTML/CSS/JavaScript (dashboard)

## Dataset

Public diabetes-iridology dataset by Moradi et al. (2018): 317 iris images from 193 subjects (150 diabetic images from 89 people, 167 healthy images from 104 people). **The dataset is not included in this repository.** Please obtain it from its original source.

## Limitations

* Small, single-source dataset; results may not transfer to other cameras, lighting or populations.
* Handcrafted features and one classifier family; deep learning was out of scope.
* Several images can come from the same subject, so a subject-level split would be a stricter evaluation than the image-level split used here.
* **Not a clinical diagnostic tool.** The explanations show what the model learned, not a proven medical link between iris patterns and diabetes.

## Future work

Larger and more diverse datasets, subject-level cross-validation, comparison with deep learning, other XAI methods alongside LIME, and review by medical professionals.

## Author

**Nurin Jessenia Mohammad Muzlini**:www.linkedin.com/in/nurinjessenia · jessymuzlini@gmail.com

## Licence

All rights reserved. Content is shown for portfolio purposes only and may not be copied, reused or redistributed without permission.

