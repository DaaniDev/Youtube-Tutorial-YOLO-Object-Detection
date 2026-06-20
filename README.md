# Ultralytics-YOLO-Object-Detection

# 🐶 YOLOv11 Dog Breed Detection using Roboflow Dataset

Train a custom YOLOv11 object detection model to identify different dog breeds using a dataset hosted on Roboflow. This project provides a complete Google Colab workflow for dataset download, model training, evaluation, and inference.

## 🚀 Overview

This repository demonstrates how to:

- Download a custom Dog Breed dataset from Roboflow
- Train a YOLOv11 model in Google Colab
- Evaluate model performance
- Run inference on images
- Export trained weights for deployment

## 📋 Prerequisites

- Google Colab Account
- Roboflow Account
- Python 3.10+
- Basic understanding of Object Detection

## 📦 Install Dependencies

```python
!pip install -q ultralytics roboflow
```

## 📥 Download Dataset from Roboflow

Replace the placeholders with your own Roboflow credentials.

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov11")
```

## 🏋️ Train YOLOv11

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data=dataset.location + "/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

## 📊 Validate Model

```python
metrics = model.val()
```

## 🔍 Perform Inference

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(
    source="test.jpg",
    conf=0.25,
    save=True
)
```

## 💾 Export Model

### ONNX

```python
model.export(format="onnx")
```

### TensorFlow Lite

```python
model.export(format="tflite")
```

## 📁 Project Structure

```text
.
├── Dog_Breed_Detection.ipynb
├── README.md
├── requirements.txt
├── dataset/
├── runs/
└── results/
```

## 📈 Results

After training, YOLOv11 automatically generates:

- Precision
- Recall
- mAP50
- mAP50-95
- Confusion Matrix
- Sample Predictions

Example Metrics:

| Metric | Value |
|----------|---------|
| Precision | 90.8% |
| Recall | 89.7% |
| mAP50 | 92.4% |
| mAP50-95 | 78.6% |

## 🎯 Sample Use Cases

- Dog Breed Identification
- Pet Monitoring Systems
- Veterinary Applications
- Smart Pet Cameras
- Animal Research Projects

## 🤝 Contributing

Feel free to fork this repository, improve the notebook, and submit pull requests.

## 📜 License

This project is licensed under the MIT License.

## ⭐ Support

If you found this project helpful, please give the repository a star.
