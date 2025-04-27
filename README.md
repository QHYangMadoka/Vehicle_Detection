# Vehicle Detection and App Deployment with YOLOv8 (UA-DETRAC)

## Overview
This project applies the YOLOv8 object detection framework to the UA-DETRAC dataset for vehicle detection, covering both model training and a lightweight desktop app deployment.

The complete pipeline includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Model training and optimization
- App deployment for quick video processing

---

## Dataset
- **UA-DETRAC**: A large-scale real-world dataset for vehicle detection in traffic videos.
- Structure:
  - `images/`: Extracted video frames.
  - `Test-Annotations/`: Original XML labels.
  - `labels/`: YOLO-format labels generated in preprocessing.

---

## Model Training Summary
- Framework: **Ultralytics YOLOv8**
- Steps:
  - XML annotations parsed and converted into YOLO `[class, x_center, y_center, width, height]` format.
  - Ignored regions are filtered.
  - Training/Validation/Test splits created.
- Evaluation Metrics:
  - **mAP**, **Precision**, **Recall**, **F1-Score**
- Optimization:
  - Hyperparameter tuning (batch size, learning rate, data augmentation like mosaic and HSV shift).
  - Post-processing predictions to remove overlaps and filter low-confidence detections.

---

## Key Results

| Version | Precision | Recall | F1-Score | F1 Variance |
| :------ | :--------- | :------ | :-------- | :---------- |
| Before Optimization | 0.6329 | 0.6390 | 0.6260 | 0.0213 |
| After Filtering | 0.7815 | 0.6172 | 0.6884 | 0.0166 |

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




## App
The app supports:
- Selecting a video for frame-by-frame vehicle detection
- Exporting a new video with bounding boxes and labels
- Viewing processing progress with a progress bar
- Canceling detection if needed

---

## Project Structure

| Stage | File | Description |
| :---- | :--- | :---------- |
| 1 | `app.py` | Main application source code |
| 2 | `models/best.pt` | Pretrained YOLOv8 model weights |
| 3 | `requirements.txt` | Python environment requirements |
| 4 | `pic/` | Interface screenshots and demo videos |

---

## Environment Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Required Python packages:
- ultralytics
- opencv-python
- PySide6
- tqdm
- torch (if not auto-installed)

---

## How to Run

### 1. Launch the Application

```bash
python app.py
```

### 2. User Interface

| Button | Function |
| :----- | :-------- |
| **Select** | Choose a video file for detection |
| **Export** | Generate processed video with bounding boxes |
| **Cancel** | Abort processing if needed |

### 3. Processing Flow

- Select a video file (e.g., `.mp4`)
- The app extracts frames, runs YOLOv8 detection, and saves labels in `temp/labels/`
- After prediction, click **Export** to generate a new video with drawn bounding boxes and labels
- Output video is saved under `runs/processed_videos/`

### 4. Output Directory

| Folder | Description |
| :----- | :----------- |
| `temp/frames/` | Temporary extracted frames |
| `temp/labels/` | YOLO format detection results |
| `runs/processed_videos/` | Final annotated video output |

---

## Interface and Demos

### Interface

![Interface](pic/interface.png)

### Original Video Demo

![Original Video](pic/video.gif)

### Processed Video with Detections

![Processed Video](pic/processed_video.gif)

---

## Notes

- All temporary frames and labels will be automatically cleared after video export.
- If the video has different resolutions between frames, the app dynamically adapts frame size.
- Canceling during export stops the video generation safely.

---

## Acknowledgements
- [UA-DETRAC Dataset](https://detrac-db.rit.albany.edu/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

