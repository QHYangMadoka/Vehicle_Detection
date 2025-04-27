import sys, os, shutil, cv2, subprocess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QMessageBox, QProgressBar, QLabel
)
from PySide6.QtCore import QRect, QThread, Signal, QObject
from ultralytics import YOLO
from glob import glob
import numpy as np

# Configuration paths
FRAME_DIR = "temp/frames"
LABEL_DIR = "temp/labels"
OUTPUT_DIR = "runs/processed_videos"
MODEL_PATH = "models/best.pt"

# Class definitions
CLASS_NAMES = ['car', 'bus', 'van', 'others']
COLORS = [(255, 0, 0), (0, 255, 255), (0, 128, 255), (255, 0, 255)]

# Load YOLO model
model = YOLO(MODEL_PATH)

cancel_flag = False

def extract_frames_and_predict(video_path, progress_callback):
    global cancel_flag
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)
    shutil.rmtree(FRAME_DIR)
    shutil.rmtree(LABEL_DIR)
    os.makedirs(FRAME_DIR)
    os.makedirs(LABEL_DIR)

    cap = cv2.VideoCapture(video_path)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    while True:
        if cancel_flag:
            break
        ret, frame = cap.read()
        if not ret:
            break

        img_name = f"{basename}_img{frame_id:04d}.jpg"
        save_path = os.path.join(FRAME_DIR, img_name)
        cv2.imwrite(save_path, frame)

        results = model.predict(frame, conf=0.25, iou=0.4)
        result = results[0]
        label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))
        with open(label_path, 'w') as f:
            for box, score, cls in zip(result.boxes.xywh, result.boxes.conf, result.boxes.cls):
                x, y, w_box, h_box = box.tolist()
                x /= frame.shape[1]
                y /= frame.shape[0]
                w_box /= frame.shape[1]
                h_box /= frame.shape[0]
                f.write(f"{int(cls)} {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")

        frame_id += 1
        progress_callback(int((frame_id / total_frames) * 60))

    cap.release()




class ExportThread(QThread):
    progress = Signal(int)
    finished = Signal(bool)

    def __init__(self, video_name):
        super().__init__()
        self.video_name = video_name

    def run(self):
        global cancel_flag
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        image_files = sorted(glob(os.path.join(FRAME_DIR, f"{self.video_name}_img*.jpg")))
        if not image_files:
            self.finished.emit(False)
            return

        out_path = os.path.join(OUTPUT_DIR, f"{self.video_name}.mp4")
        out = None  # Lazy initialization

        total = len(image_files)
        for i, img_path in enumerate(image_files):
            if cancel_flag:
                break
            fname = os.path.basename(img_path)
            label_path = os.path.join(LABEL_DIR, fname.replace('.jpg', '.txt'))
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            if out is None:
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        cls, x, y, w_box, h_box = map(float, line.strip().split())
                        x *= w
                        y *= h
                        w_box *= w
                        h_box *= h
                        x1 = int(x - w_box / 2)
                        y1 = int(y - h_box / 2)
                        x2 = int(x + w_box / 2)
                        y2 = int(y + h_box / 2)

                        color = COLORS[int(cls) % len(COLORS)]
                        label_name = CLASS_NAMES[int(cls)]
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        (tw, th), bl = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, y1 - th - bl), (x1 + tw, y1), color, -1)
                        cv2.putText(img, label_name, (x1, y1 - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            out.write(img)
            if os.path.exists(label_path):
                os.remove(label_path)
            self.progress.emit(60 + int((i / total) * 40))

        if out:
            out.release()
        if not cancel_flag:
            subprocess.Popen(f'explorer \"{os.path.abspath(OUTPUT_DIR)}\"')
        self.finished.emit(not cancel_flag)


class SimpleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Detection App")
        self.setGeometry(300, 300, 400, 300)

        self.progress = QProgressBar(self)
        self.progress.setGeometry(QRect(50, 80, 300, 30))
        self.progress.setValue(0)

        self.status = QLabel("Waiting...", self)
        self.status.setGeometry(QRect(50, 120, 300, 30))

        self.btn_select = QPushButton("Select", self)
        self.btn_select.setGeometry(QRect(50, 200, 90, 40))
        self.btn_select.clicked.connect(self.select_video)

        self.btn_export = QPushButton("Export", self)
        self.btn_export.setGeometry(QRect(155, 200, 90, 40))
        self.btn_export.clicked.connect(self.export_video)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setGeometry(QRect(260, 200, 90, 40))
        self.btn_cancel.clicked.connect(self.cancel_operation)

        self.video_path = None
        self.export_thread = None

    def select_video(self):
        global cancel_flag
        cancel_flag = False
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "./", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            video_name = os.path.splitext(os.path.basename(path))[0]
            self.btn_select.setEnabled(False)
            self.btn_export.setEnabled(False)
            self.progress.setValue(0)
            self.status.setText("Step 1/3: Splitting and predicting...")
            extract_frames_and_predict(path, self.progress.setValue)
            if not cancel_flag:
                QMessageBox.information(self, "Done", "Prediction complete. Click Export to continue.")
            self.btn_select.setEnabled(True)
            self.btn_export.setEnabled(True)

    def export_video(self):
        global cancel_flag
        cancel_flag = False
        if not self.video_path:
            QMessageBox.warning(self, "Error", "Please select a video first.")
            return
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.btn_select.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.status.setText("Step 2/3: Generating video...")
        self.progress.setValue(60)

        self.export_thread = ExportThread(video_name)
        self.export_thread.progress.connect(self.progress.setValue)
        self.export_thread.finished.connect(self.on_export_done)
        self.export_thread.start()

    def on_export_done(self, success):
        if success:
            self.status.setText("Step 3/3: ✔ Success!")
            self.progress.setValue(100)
        else:
            self.status.setText("Canceled.")
        self.btn_select.setEnabled(True)
        self.btn_export.setEnabled(True)

    def cancel_operation(self):
        global cancel_flag
        cancel_flag = True
        self.status.setText("Canceling...")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SimpleApp()
    window.show()
    sys.exit(app.exec())
