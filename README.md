# Vehicle Detection with YOLO on UA-DETRAC Dataset

## Overview
This project applies the YOLO (You Only Look Once) object detection framework to the UA-DETRAC dataset for vehicle detection. The complete pipeline covers **data preprocessing**, **exploratory data analysis**, **model training**, **model optimization**, and **post-optimization processing**.

---

## Project Structure

| Stage | File | Description |
| :---- | :--- | :---------- |
| 1 | `1.Data-Preprocessing.ipynb` | Parsing XML annotations, generating YOLO format labels, splitting datasets. |
| 2 | `2.Exploratory Data Analysis.ipynb` | Visualizing data distribution, object size statistics, bounding box heatmaps. |
| 3 | `3.YOLO_model_train&evaluation.ipynb` | Training YOLOv8 model, evaluating with mAP and loss metrics. |
| 4 | `4.YOLO_model_optimization.ipynb` | Fine-tuning hyperparameters (batch size, learning rate, data augmentation). |
| 5 | `5.YOLO_post_optimization.ipynb` | Post-processing predictions: filtering overlapping boxes, final evaluation. |

---

## Dataset

- **UA-DETRAC**: A large-scale challenging real-world dataset for vehicle detection from traffic videos.
- **Data Structure**:
  - `images/` – extracted frames from videos.
  - `Test-Annotations/` – original XML annotation files.
  - `labels/` – YOLO format labels generated during preprocessing.

---

## Pipeline Details

### 1. Data Preprocessing
- Parse XML annotations (`*.xml`) and extract bounding boxes.
- Ignore regions marked in XML (`<ignored_region>` tag).
- Normalize bounding boxes into YOLO `[class x_center y_center width height]` format.
- Split dataset into **train**, **val**, and **test** sets.

### 2. Exploratory Data Analysis (EDA)
- Plot number of vehicles per frame.
- Visualize bounding box size and aspect ratio distribution.
- Heatmap of object locations to analyze camera angles and scene bias.
- Detect outliers in label data.

### 3. YOLO Model Training & Evaluation
- Model: **YOLOv8** (Ultralytics).
- Trained using original images and generated YOLO labels.
- Evaluation Metrics:
  - **mAP (mean Average Precision)**
  - **Precision, Recall**
  - **Loss curves** (classification loss, localization loss, objectness loss).
- Save best checkpoint based on validation mAP.

### 4. YOLO Model Optimization
- Hyperparameter tuning:
  - Learning rate scheduling.
  - Anchor box optimization.
  - Data augmentation (mosaic, mixup, HSV shift).
- Retrain model with optimized settings.
- Re-evaluate mAP improvements.

### 5. Post Optimization
- Apply prediction filtering:
  - Remove predictions overlapping with ignored regions.
  - Filter small or low-confidence bounding boxes.
- Final evaluation on the cleaned test set.
- Save final prediction results.

---

## Results

| Version | Precision Mean | Recall Mean | F1 Mean | F1 Variance |
| :------ | :------------- | :---------- | :------ | :---------- |
| Before Optimization | 0.6329 | 0.6390 | 0.6260 | 0.0213 |
| After Filtering | 0.7815 | 0.6172 | 0.6884 | 0.0166 |

_(Tip: Fill the table with your real experimental results.)_

---

## Environment

- Python 3.8+
- YOLOv8 (Ultralytics)
- OpenCV
- lxml
- Matplotlib, Seaborn

Install requirements:
```bash
pip install -r requirements.txt
```

---

## Acknowledgements
- [UA-DETRAC Dataset](https://detrac-db.rit.albany.edu/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

