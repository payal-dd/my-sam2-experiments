# 🎯 Object Segmentation using Grounded SAM 2

A computer vision project that detects and segments objects in images using **Grounding DINO** + **SAM 2 (Segment Anything Model 2)** — running locally on Windows with CUDA GPU acceleration.

---

## 📌 What This Project Does

- Detects objects in any image using **text prompts** (e.g. "plant. couch. cushion.")
- Segments each detected object with a **colored mask overlay**
- Runs fully locally on your GPU — no cloud API needed
- Interactive exploration via **JupyterLab notebook**

---

## 🖼️ Demo Output

| Input Image | Segmented Output |
|---|---|
| Living room photo | 16 objects detected — couch, plants, cushions, table |

> Detected objects with confidence scores using Grounding DINO + SAM 2.1 Large model on NVIDIA RTX 3050 6GB

---

## 🛠️ Tech Stack

| Tool | Version |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| SAM 2 | 2.1 (hiera_large) |
| Grounding DINO | tiny (HuggingFace) |
| Transformers | 5.7.0 |
| JupyterLab | Latest |
| GPU | NVIDIA GeForce RTX 3050 6GB |
| OS | Windows 11 |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/payal-dd/my-sam2-experiments.git
cd my-sam2-experiments
```

### 2. Create and activate virtual environment
```bash
python -m venv gsam2_env
gsam2_env\Scripts\activate        # Windows
```

### 3. Install PyTorch with CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install SAM 2 dependencies
```bash
pip install -e .
pip install supervision transformers pycocotools jupyterlab
```

### 5. Download SAM 2 checkpoint
```bash
cd checkpoints
curl -L -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ..
```

---

## 🚀 How to Run

### Launch JupyterLab
```bash
# Register the kernel
pip install ipykernel
python -m ipykernel install --user --name=gsam2_env --display-name "SAM2 (gsam2_env)"

# Launch JupyterLab
jupyter lab
```

### In JupyterLab
1. Open `Object_Segmentation.ipynb`
2. Select kernel **SAM2 (gsam2_env)**
3. Change the image path and text prompt to your own image:
```python
image = Image.open("your_image.jpg").convert("RGB")
TEXT_PROMPT = "person. car. dog."   # match objects in your image
```
4. Run all cells with **Shift+Enter**

---

## 📁 Project Structure

```
my-sam2-experiments/
│
├── Object_Segmentation.ipynb    # Main notebook
├── checkpoints/                 # SAM 2 model weights
├── outputs/                     # Segmentation results
│   └── grounded_sam2_hf_demo/
│       ├── groundingdino_annotated_image.jpg
│       └── grounded_sam2_annotated_image_with_mask.jpg
├── sam2/                        # SAM 2 source code
└── README.md
```

---

## 📖 How It Works

```
Your Image
    │
    ▼
Grounding DINO  ──►  Detects objects using text prompt
    │                (returns bounding boxes + labels)
    ▼
SAM 2 Predictor ──►  Segments each bounding box
    │                (returns precise pixel masks)
    ▼
Visualization   ──►  Overlays colored masks + labels
```

---

## 📝 Usage Example

```python
# Load your image
image = Image.open("living_room.jpg").convert("RGB")

# Detect objects with text prompt
TEXT_PROMPT = "plant. couch. table. cushion."

# Run detection + segmentation
# → Detected 16 objects: couch(0.80), plant(0.63), cushion(0.60)...
```

---

## ⚠️ Requirements

- NVIDIA GPU with CUDA support (tested on RTX 3050 6GB)
- CUDA Toolkit 12.1
- Windows 10/11 or Linux
- Python 3.10
- ~4GB free VRAM for SAM 2 Large model

> For faster inference use `sam2.1_hiera_tiny.pt` (148MB) instead of large (856MB)

---

## 🙏 Credits

- [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) by IDEA-Research
- [SAM 2](https://github.com/facebookresearch/segment-anything-2) by Meta AI
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) by IDEA-Research

---

## 👩‍💻 Author

**Payal** — [@payal-dd](https://github.com/payal-dd)
