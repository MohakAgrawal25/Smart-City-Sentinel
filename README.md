# 🌆 SmartVision Urban

### AI-Powered Smart City Infrastructure Monitoring & Geo-Tagged Fault Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/DeepLearning-ComputerVision-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Framework-Gradio-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Map-Leaflet.js-brightgreen?style=for-the-badge">
</p>

---

# 🚀 Overview

**SmartVision Urban** is an advanced **multi-model deep learning framework** designed for automated smart city infrastructure monitoring using **YOLOv8 object detection models**.

The system intelligently detects multiple categories of urban infrastructure faults from a single image and maps detected issues on an interactive geo-tagged city map.

The framework combines:

✅ Deep Learning
✅ Computer Vision
✅ Geo-Spatial Intelligence
✅ Smart City Automation
✅ Real-Time Fault Monitoring

---

# 🎯 Problem Statement

Modern cities face serious challenges in maintaining infrastructure such as:

* 🛣️ Potholes & Road Cracks
* 🌳 Fallen Trees
* 🚦 Damaged Traffic Signs
* 💡 Street Light Failures
* ⚡ Damaged Electric Poles
* 🗑️ Garbage Accumulation

Traditional manual inspection is:

❌ Slow
❌ Expensive
❌ Error-Prone
❌ Non-scalable

SmartVision Urban automates this entire process using AI-powered infrastructure inspection.

---

# 🧠 Key Features

## ✅ Multi-Model YOLO Architecture

Six independently trained YOLOv8 models specialized for different urban fault categories.

## ✅ Unified Detection Pipeline

Single input image → processed through all six models sequentially.

## ✅ Geo-Tagging Support

Extracts latitude & longitude directly from image filenames.

Example:

```bash
img_21.1234,79.0823.jpg
```

## ✅ Interactive Fault Mapping

Detected faults automatically appear on a city map using Leaflet.js.

## ✅ Modular & Scalable

New detection models can be added easily without modifying the existing system.

## ✅ Real-Time Smart City Monitoring

Can be integrated with:

* CCTV Cameras
* Drones
* Mobile Applications
* Citizen Reporting Systems

---

# 🏗️ System Architecture

```text
                    +------------------+
                    |  Input Image     |
                    +------------------+
                              |
                              v
        ------------------------------------------------
        |              YOLOv8 Detection Pipeline        |
        ------------------------------------------------
          |         |         |        |       |       
          v         v         v        v       v
     Fallen      Damaged   Pothole  Street  Electric
      Tree         Sign     Crack    Light    Pole
       Model       Model     Model    Model    Model
                              |
                              v
                        Garbage Model
                              |
                              v
                  +----------------------+
                  | Detection Aggregator |
                  +----------------------+
                              |
                              v
                +--------------------------+
                | Geo-Tagged Fault Mapping |
                +--------------------------+
                              |
                              v
                     Interactive City Map
```

---

# 🔍 Detection Categories

| Model   | Detection Task                  |
| ------- | ------------------------------- |
| Model 1 | Fallen Tree Detection           |
| Model 2 | Damaged Road Sign Detection     |
| Model 3 | Pothole & Road Crack Detection  |
| Model 4 | Street Light Status Detection   |
| Model 5 | Damaged Electric Pole Detection |
| Model 6 | Garbage Detection               |

---

# 📂 Dataset Information

Datasets were collected from:

* Roboflow Universe
* Kaggle
* Custom Field-Collected Images

All datasets were annotated in YOLO format.

---

# 📊 Dataset Statistics

| Model                 | Train | Validation | Test | Classes |
| --------------------- | ----- | ---------- | ---- | ------- |
| Fallen Tree           | 8500  | 982        | 1632 | 1       |
| Damaged Road Sign     | 4863  | 914        | 136  | 1       |
| Pothole & Crack       | 5667  | 1219       | 369  | 2       |
| Street Light          | 2499  | 262        | 120  | 1       |
| Damaged Electric Pole | 7271  | 513        | 328  | 1       |
| Garbage Detection     | 3133  | 660        | 337  | 1       |

---

# 🧪 Technologies Used

| Technology     | Purpose             |
| -------------- | ------------------- |
| Python 3.10    | Core Programming    |
| YOLOv8         | Object Detection    |
| PyTorch        | Deep Learning       |
| OpenCV         | Image Processing    |
| Gradio         | User Interface      |
| Leaflet.js     | Interactive Mapping |
| OpenStreetMap  | Map Tiles           |
| Albumentations | Data Augmentation   |

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/SmartVision-Urban.git
cd SmartVision-Urban
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Required Libraries

```txt
ultralytics
opencv-python
torch
torchvision
numpy
pandas
gradio
folium
leaflet
albumentations
matplotlib
```

---

# ▶️ Running the Project

```bash
python app.py
```

---

# 🗺️ Geo-Tagging Format

Image filenames must follow this format:

```bash
img_latitude,longitude.jpg
```

Example:

```bash
img_21.1458,79.0882.jpg
```

The system automatically:

✅ Extracts GPS coordinates
✅ Detects infrastructure faults
✅ Places markers on interactive maps

---

# 📈 Performance Metrics

| Model                 | Precision | Recall | mAP@0.5 |
| --------------------- | --------- | ------ | ------- |
| Fallen Tree           | 0.8750    | 0.8198 | 0.8980  |
| Damaged Road Sign     | 0.9709    | 0.9657 | 0.9884  |
| Pothole & Crack       | 0.7377    | 0.5733 | 0.5999  |
| Street Light          | 0.6921    | 0.6474 | 0.6693  |
| Damaged Electric Pole | 0.8564    | 0.7954 | 0.8568  |
| Garbage Detection     | 0.6721    | 0.4628 | 0.5496  |

---

# 📸 Sample Workflow

```text
Input Image
     ↓
YOLOv8 Detection
     ↓
Bounding Box Predictions
     ↓
Geo Coordinate Extraction
     ↓
Interactive Map Visualization
```

---

# 🌍 Real-World Applications

## 🏙️ Municipal Smart City Systems

Automated urban fault monitoring.

## 🚓 Smart Surveillance

Integration with CCTV infrastructure.

## 🚁 Drone-Based Inspection

Large-scale infrastructure analysis using UAVs.

## 🛣️ Road Maintenance

Continuous pothole & crack monitoring.

## ⚡ Utility Infrastructure Monitoring

Electric pole and street light inspection.

## ♻️ Waste Management

Garbage hotspot detection & route optimization.

---

# 🔥 Novel Contributions

✅ Multi-model YOLO pipeline
✅ Geo-tagged fault detection
✅ Interactive smart city mapping
✅ Domain-specific model specialization
✅ Real-time scalable monitoring system

---

# 🚀 Future Improvements

* EXIF Metadata-based GPS extraction
* Parallel model inference
* Edge AI deployment
* Real-time video stream analysis
* GIS integration
* Automated maintenance ticket generation
* Mobile app deployment

---

# 📁 Project Structure

```bash
SmartVision-Urban/
│
├── models/
│   ├── fallen_tree.pt
│   ├── damaged_sign.pt
│   ├── pothole_crack.pt
│   ├── street_light.pt
│   ├── damaged_pole.pt
│   └── garbage.pt
│
├── datasets/
│
├── outputs/
│
├── maps/
│
├── app.py
├── requirements.txt
├── README.md
└── utils.py
```

---

# 📚 Research Paper

This repository is based on the research paper:

### **SmartVision Urban: A Multi-Model YOLO-Based Deep Learning Framework for Automated Smart City Infrastructure Monitoring and Geo-Tagged Fault Detection**

Authors:

* Mohak Agrawal
* Neel Futane

---

# 👨‍💻 Authors

## Mohak Agrawal

AI/ML Developer | Deep Learning Enthusiast

## Neel Futane

Computer Vision Researcher

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve the project:

1. Fork the repository
2. Create a new branch
3. Commit changes
4. Push your branch
5. Open a Pull Request

---

# ⭐ Support

If you like this project:

🌟 Star the repository
🍴 Fork the project
📢 Share with others

---

# 📜 License

This project is licensed under the MIT License.

---

# 💡 Quote

> “Smart cities need smart monitoring — SmartVision Urban brings AI-powered infrastructure intelligence to reality.”

---

# 🔗 GitHub Repository

Replace with your actual repository link:

```bash
https://github.com/your-username/SmartVision-Urban
```

---

# 🏁 Final Output

✅ AI-Based Urban Fault Detection
✅ Smart City Monitoring
✅ Geo-Tagged Infrastructure Intelligence
✅ Real-Time YOLOv8 Detection System

---

