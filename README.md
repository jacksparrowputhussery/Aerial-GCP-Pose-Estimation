# Ground Control Point (GCP) Detection using YOLO11

## Overview

This project implements an end-to-end Ground Control Point (GCP) detection pipeline using YOLO11 for aerial drone imagery.

The objective is to automatically:

* Detect GCP markers in high-resolution drone images.
* Classify the marker shape.
* Predict the center coordinates of the marker.
* Export predictions in the required JSON format.

Supported marker types:

* Cross
* Square
* L-Shape

The model is trained using point annotations provided in `gcp_marks.json`.

---

# Problem Statement

The provided dataset contains:

* High-resolution aerial images (~2000×1500 pixels)
* Marker center coordinates
* Marker shape labels

Example annotation:

```json
{
  "DJI_0001.JPG": {
    "mark": {
      "x": 1054.2,
      "y": 742.8
    },
    "verified_shape": "Cross"
  }
}
```

Unlike a standard object detection dataset, no bounding boxes are provided.

YOLO requires bounding boxes for training.

Therefore, bounding boxes are generated around each annotated center point.

---

# Architecture

```text
gcp_marks.json
       │
       ▼
Center Point Annotation
       │
       ▼
48×48 Bounding Box Generation
       │
       ▼
YOLO Dataset Creation
       │
       ▼
Train / Validation Split
       │
       ▼
YOLO11s Training
       │
       ▼
best.pt
       │
       ▼
Inference
       │
       ▼
Bounding Box Prediction
       │
       ▼
Center Coordinate Extraction
       │
       ▼
JSON Output
```

---

# Dataset Preparation

## Original Dataset

```text
train_dataset/
│
├── gcp_marks.json
│
├── scout_971/
├── GCP-11/
├── ...
```

Images may exist inside nested directories.

A recursive search is used:

```python
for root, _, files in os.walk(TRAIN_ROOT):
    ...
```

to locate every image in the dataset.

---

# Bounding Box Generation

The dataset only contains center coordinates.

Each center point is converted into a YOLO bounding box.

Configuration:

```python
BOX_SIZE = 48
```

Generated box:

```text
           48 px
     ┌─────────────┐
     │             │
     │      +      │
     │             │
     └─────────────┘
```

where `+` represents the annotated center.

---

# Why BOX_SIZE = 48?

The marker occupies a relatively small area in the aerial image.

48 pixels was selected because it:

* Covers the complete marker.
* Minimizes surrounding background.
* Produces stable YOLO labels.
* Works consistently across all marker shapes.

Tradeoff:

| Small Box           | Large Box            |
| ------------------- | -------------------- |
| Better localization | More context         |
| May crop marker     | Less accurate center |

48×48 provided the best balance.

---

# Class Mapping

```python
CLASS_MAP = {
    "Cross": 0,
    "Square": 1,
    "L-Shape": 2
}
```

---

# Train / Validation Split

Dataset split:

```text
80% Training
20% Validation
```

Using stratified sampling:

```python
train_test_split(
    test_size=0.2,
    stratify=labels
)
```

This preserves class balance across splits.

---

# YOLO Dataset Structure

Generated structure:

```text
gcp_yolo/
│
├── dataset.yaml
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
```

---

# YOLO Configuration

## dataset.yaml

```yaml
path: /content/gcp_yolo

train: images/train
val: images/val

names:
  0: Cross
  1: Square
  2: L-Shape
```

---

# Model Architecture

Model:

```python
YOLO("yolo11s.pt")
```

YOLO11s was selected because it offers:

* Good small-object detection
* Fast training
* Low memory consumption
* Suitable for Colab T4

---

# Training Environment

## Hardware

Google Colab

GPU:

```text
Tesla T4
16 GB VRAM
```

Environment:

```text
Python 3.12
PyTorch 2.11
CUDA 12.8
Ultralytics 8.x
```

---

# Training Configuration

```python
IMAGE_SIZE = 1536
BATCH_SIZE = 8
EPOCHS = 100
YOLO_MODEL = "yolo11s.pt"
```

Training:

```python
model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=1536,
    batch=8,
    device=0,
    cache=True,
    amp=True
)
```

---

# Why IMAGE_SIZE = 1536?

Original aerial images are approximately:

```text
2000 × 1500
```

Small GCP markers become difficult to detect when resized aggressively.

Using:

```text
1536 × 1536
```

helps preserve marker detail.

Advantages:

* Better small-object detection
* Higher recall
* Improved localization accuracy

Compared with:

```text
640 × 640
```

which significantly reduces marker size.

---

# Why BATCH_SIZE = 8?

Training is performed on a Tesla T4.

At:

```text
1536 × 1536
```

resolution, GPU memory usage becomes significant.

Batch size 8:

* Fits comfortably within 16 GB VRAM
* Provides stable gradients
* Maximizes GPU utilization

---

# Why 100 Epochs?

```python
EPOCHS = 100
```

Reasons:

* Small object detection generally converges slower.
* Allows sufficient exposure to all classes.
* Produces stable validation metrics.

---



Benefits:

* Better generalization
* Increased robustness
* Improved performance on varying aerial conditions

---

# Inference Pipeline

Load trained model:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

Run inference:

```python
results = model(
    image_path,
    conf=0.25
)
```

Extract best detection:

```python
boxes = results[0].boxes

best_idx = boxes.conf.argmax()

x1, y1, x2, y2 = boxes.xyxy[best_idx]
```

Compute center:

```python
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

Map class:

```python
shape = names[class_id]
```

---

# Output Format

Prediction output:

```json
{
  "image.jpg": {
    "mark": {
      "x": 1054.2,
      "y": 742.8
    },
    "verified_shape": "Cross"
  }
}
```

---

# Handling Missing Detections

If no marker is found:

```json
{
  "mark": {
    "x": -1,
    "y": -1
  }
}
```

This explicitly indicates detection failure.

---

# Advantages of YOLO over Sliding Window Classification

Traditional approach:

```text
Large Image
    │
    ▼
Sliding Window
    │
    ▼
EfficientNet
```

Problems:

* Slow
* Redundant computation
* Multiple overlapping predictions

YOLO approach:

```text
Large Image
    │
    ▼
YOLO11
    │
    ├── Location
    └── Shape
```

Advantages:

* Single forward pass
* Faster inference
* Better localization
* Scalable to larger datasets

---

# Results

The final model can:

* Detect GCP markers from full aerial images.
* Predict marker centers.
* Classify Cross, Square and L-Shape markers.
* Generate JSON outputs directly compatible with the evaluation pipeline.

---
