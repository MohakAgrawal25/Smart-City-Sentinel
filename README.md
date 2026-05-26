```markdown
# 🌆 SmartVision Urban

**A Multi-Model YOLO-Based Deep Learning Framework for Automated Smart City Infrastructure Monitoring and Geo-Tagged Fault Detection**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio App](https://img.shields.io/badge/Demo-Gradio-blueviolet.svg)](https://gradio.app/)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Detection Models](#-detection-models)
- [Dataset & Training](#-dataset--training)
- [Performance Metrics](#-performance-metrics)
- [Installation](#-installation)
- [Usage](#-usage)
- [Geo-Tagging Convention](#-geo-tagging-convention)
- [Project Structure](#-project-structure)
- [Applications](#-applications)
- [Limitations & Future Work](#-limitations--future-work)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

**SmartVision Urban** is a modular, production-ready deep learning pipeline that simultaneously detects six categories of urban infrastructure faults from street-level imagery and maps them onto an interactive geospatial interface.

The system deploys **six independently trained YOLOv8 models**, each specialized in a distinct urban anomaly:
- 🌳 Fallen Trees  
- 🛑 Damaged Road Signs  
- 🕳️ Potholes & Road Cracks  
- 💡 Street Light Operational Status  
- ⚡ Damaged Electric Poles  
- 🗑️ Garbage Accumulation  

Detected faults are automatically geo‑referenced using GPS coordinates embedded in the image filename and displayed as markers on a dynamic **Leaflet.js map** – enabling municipal authorities to visualize, prioritize, and act on infrastructure issues with minimal manual effort.

---

## ✨ Key Features

- **Six Specialist Models** – Each YOLOv8 model is optimized for one fault category, providing higher accuracy than a single multi‑class model.
- **Unified Inference Pipeline** – One image is sequentially passed through all six models; results are aggregated and visualized with bounding boxes and confidence scores.
- **Geo‑Tagged Map Interface** – GPS coordinates are extracted from image filenames and faults are automatically plotted on an OpenStreetMap‑based map.
- **Modular Architecture** – Add, remove, or update individual detection models without disrupting the entire pipeline.
- **Gradio Web Demo** – User‑friendly interface to upload images, run inference, and view both detections and the fault map.
- **Optimized for Real‑World Data** – Trained on diverse datasets collected across different weather conditions, times of day, and geographic locations.

---

## 🧱 System Architecture

![Architecture Diagram](docs/architecture.png)  
*(Figure: End‑to‑end flow – image ingestion → multi‑model inference → result aggregation → geo‑visualization)*

The pipeline consists of three major stages:
1. **Inference** – The input image passes sequentially through all six YOLOv8 models. Each model returns its own bounding boxes, class labels, and confidence scores.
2. **Aggregation** – All detections are merged onto the original image; distinct colours are assigned per fault category.
3. **Geo‑Mapping** – Latitude and longitude are parsed from the filename (format `img_lat,long.jpg`) and a corresponding marker is placed on the interactive map.

---

## 🧠 Detection Models

All models are built on **Ultralytics YOLOv8** with anchor‑free decoupled heads, CSP‑based backbone, and PAFNet neck. Each model was trained independently with hyperparameters tuned for its specific task.

| # | Model | Task | Classes | Train Images | mAP@0.5 |
|---|-------|------|---------|---------------|---------|
| 1 | **Fallen Tree Detector** | Detect trees fallen across roads/sidewalks | `fallen_tree` | 8,500 | 0.898 |
| 2 | **Damaged Road Sign Detector** | Bent, broken, missing, or vandalised signs | `damaged_sign` | 4,863 | 0.988 |
| 3 | **Pothole & Crack Detector** | Potholes, longitudinal/transverse/alligator cracks | `pothole`, `crack` | 5,667 | 0.600 |
| 4 | **Street Light Status Detector** | Classify street light as ON or OFF | `street_light` | 2,499 | 0.669 |
| 5 | **Damaged Electric Pole Detector** | Leaning, cracked, or structurally compromised poles | `damaged_pole` | 7,271 | 0.857 |
| 6 | **Garbage Detector** | Overflowing bins, litter piles, illegal dumping | `garbage` | 3,133 | 0.550 |

All performance metrics were computed on held‑out test sets.

---

## 📊 Dataset & Training

### Data Sources
Datasets were curated from **Roboflow Universe**, **Kaggle**, and custom field collections. Images were annotated in YOLO format and split into training, validation, and test sets (approx. 75:15:10).

### Augmentations
- Random horizontal/vertical flip (p=0.5)
- Random rotation (±15°)
- Mosaic augmentation (4‑image merging)
- HSV jitter (hue ±0.015, saturation ±0.7, value ±0.4)
- Random scaling & translation

### Hyperparameters
- Optimizer: AdamW (lr=0.001, cosine schedule)
- Momentum: 0.937, weight decay: 0.0005
- Batch size: 16, epochs: 100 (early stopping patience 20)
- Input resolution: 640×640
- NMS IoU threshold: 0.45, confidence threshold: 0.25

Training was performed on an **NVIDIA RTX 3060 12GB** GPU with PyTorch 2.0.1.

---

## 📈 Performance Metrics

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|---------------|
| Fallen Tree | 0.875 | 0.820 | 0.898 | 0.545 |
| Damaged Road Sign | 0.971 | 0.966 | 0.988 | 0.776 |
| Pothole & Crack | 0.738 | 0.573 | 0.600 | 0.327 |
| Street Light Status | 0.692 | 0.647 | 0.669 | 0.346 |
| Damaged Electric Pole | 0.856 | 0.795 | 0.857 | 0.628 |
| Garbage Detection | 0.672 | 0.463 | 0.550 | 0.300 |

*Note: Metrics were computed on test sets; the street light and garbage models can be further improved with larger and more diverse datasets.*

---

## 💻 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/smartvision-urban.git
cd smartvision-urban
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Dependencies include**:
- `torch>=2.0`
- `ultralytics>=8.0.196`
- `gradio`
- `opencv-python`
- `leafmap` / `folium` (map rendering)
- `exifread` (optional, for EXIF GPS extraction)
- `matplotlib`, `numpy`, `pandas`

4. **Download pre‑trained weights**
The model weights are stored in the `weights/` directory. Run the download script or manually place your `.pt` files:
```bash
python download_weights.py
```
*Alternatively, use the links provided in `weights/README.md`.*

---

## 🚀 Usage

### Gradio Web Application
Launch the interactive interface:
```bash
python app.py
```
Open the local URL (usually `http://127.0.0.1:7860`) in your browser.  
You can upload an image, see bounding boxes and confidence scores, and view fault markers on the map.

### Command‑Line Inference
Run all models on a single image and save the annotated result:
```bash
python detect.py --image path/to/img_21.1234,79.0823.jpg --output results/
```
Optional flags:
- `--models` : comma‑separated list of model names (e.g., `tree,sign`) to run only specific detectors.
- `--conf` : confidence threshold (default 0.25).
- `--save-map` : export a standalone HTML map with fault markers.

### Example
```bash
python detect.py --image samples/img_19.0760,72.8777.jpg --conf 0.3
```

The output will show all detected faults and generate a file `results/img_19.0760,72.8777_annotated.jpg` along with a map snippet.

---

## 🗺️ Geo‑Tagging Convention

SmartVision Urban extracts GPS coordinates directly from the **image filename** using the format:
```
img_latitude,longitude.jpg
```
Example: `img_28.6139,77.2090.jpg`  ← (New Delhi, India)

If the filename does not match this pattern, no geo‑marker is placed (the image will still be processed for faults).

Future versions will support automatic extraction from EXIF metadata.

---

## 📁 Project Structure

```
smartvision-urban/
├── app.py                  # Gradio web application
├── detect.py               # Command‑line inference script
├── download_weights.py     # Script to download pretrained weights
├── requirements.txt
├── models/                 # YOLO model definitions (if custom)
├── weights/                # Pretrained .pt weight files
│   ├── fallen_tree.pt
│   ├── damaged_sign.pt
│   ├── pothole_crack.pt
│   ├── street_light.pt
│   ├── damaged_pole.pt
│   └── garbage.pt
├── utils/
│   ├── geo_utils.py        # Coordinate parsing & map generation
│   └── visualization.py    # Bounding box drawing, colour mapping
├── datasets/               # Dataset split references (optional)
├── samples/                # Example images for testing
├── results/                # Output directory for annotated images & maps
├── docs/                   # Documentation, architecture diagram
└── README.md
```

---

## 🏙️ Applications

- **Municipal Corporations** – Automated screening of citizen‑reported images or CCTV feeds.
- **Road & Highway Authorities** – Continuous pavement condition monitoring using dashcams.
- **Electricity Distribution Companies** – Post‑storm aerial/ground inspection of poles and street lights.
- **Solid Waste Management** – Identify garbage hotspots and optimise collection routes.
- **Smart City Dashboards** – Integration with existing Geographic Information Systems (GIS).

---

## 🚧 Limitations & Future Work

- **Sequential processing** currently leads to higher total inference time; parallelisation and model distillation are planned.
- **Geo‑tagging** relies on filenames – automatic EXIF GPS extraction will be added.
- Dataset diversity can be expanded to more countries and climate zones.
- **Video stream support** and temporal anomaly tracking are under development.
- Direct integration with municipal **work‑order systems** is a future goal.

---

## 🤝 Contributing

Contributions are welcome! Please follow the standard GitHub workflow:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

See `CONTRIBUTING.md` for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use SmartVision Urban in your research or project, please cite our paper:

```bibtex
@article{agrawal2024smartvision,
  title     = {SmartVision Urban: A Multi-Model YOLO-Based Deep Learning Framework for Automated Smart City Infrastructure Monitoring and Geo-Tagged Fault Detection},
  author    = {Mohak Agrawal and Neel Futane},
  journal   = {arXiv preprint},
  year      = {2024},
  note      = {Under review}
}
```

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation.
- [Roboflow](https://roboflow.com/) and [Kaggle](https://kaggle.com/) for providing open datasets.
- All contributors of open‑source libraries used in this project.

---

<p align="center">
  <sub>Built with ❤️ for smarter, safer cities.</sub>
</p>
```
