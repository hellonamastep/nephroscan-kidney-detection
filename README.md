# 🫘 NephroScan — AI Kidney Stone Detection

> **Live Demo →** [huggingface.co/spaces/YOUR_USERNAME/nephroscan](https://huggingface.co/spaces/YOUR_USERNAME/nephroscan)

An end-to-end deep learning project that classifies kidney CT scans into
**Normal · Stone · Cyst · Tumor** using MobileNetV2 transfer learning.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey?style=flat-square)
![Accuracy](https://img.shields.io/badge/Val_Accuracy-~96%25-green?style=flat-square)

---

## 🎥 Demo

| Upload CT Scan | Detection Result |
|---|---|
| Drag & drop any kidney CT image | Instant classification with confidence % |

---

## 🧠 Model Architecture

```
Input (224×224×3 CT Scan)
        ↓
MobileNetV2 (ImageNet pretrained)
  Phase 1 → base frozen, head trains
  Phase 2 → last 30 layers fine-tuned
        ↓
GlobalAveragePooling2D
        ↓
Dense(512, ReLU) + Dropout(0.4)
Dense(256, ReLU) + Dropout(0.3)
        ↓
Dense(4, Softmax) → [Cyst · Normal · Stone · Tumor]
```

**Training — 2-phase transfer learning:**
- Phase 1: Train classification head (base frozen) — 20 epochs
- Phase 2: Fine-tune last 30 MobileNetV2 layers — 15 epochs
- Best val accuracy: **~96%**

---

## 📊 Dataset

**CT Kidney Dataset** — 12,446 CT images from Kaggle

| Class | Images | Description |
|-------|--------|-------------|
| Normal | 5,077 | Healthy kidneys |
| Cyst | 3,709 | Fluid-filled sacs |
| Stone | 1,377 | Mineral deposits |
| Tumor | 2,283 | Renal masses |

Source: [Kaggle — CT Kidney Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/nephroscan-kidney-detection
cd nephroscan-kidney-detection

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# Place model files in model/ folder:
#   model/nephroscan_final.keras
#   model/class_indices.json

python app.py
# Open http://localhost:5000
```

---

## 📁 Project Structure

```
nephroscan-kidney-detection/
├── app.py                  # Flask backend
├── Dockerfile              # HF Spaces deployment
├── requirements.txt
├── .gitignore
├── README.md
├── templates/
│   └── index.html          # Frontend UI
└── model/
    └── class_indices.json  # Class label mapping
```

---

## 🔌 API

```
POST /predict     Upload CT scan → returns JSON prediction
GET  /health      Model health check
GET  /            Web UI
```

**Sample Response:**
```json
{
  "prediction": "Stone",
  "confidence": 97.4,
  "probabilities": { "Stone": 97.4, "Normal": 1.8, "Cyst": 0.6, "Tumor": 0.2 },
  "severity": "DETECTED",
  "description": "Kidney stones are hard mineral deposits...",
  "recommendation": "Consult a urologist promptly..."
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | TensorFlow 2.x / Keras |
| Base CNN | MobileNetV2 (ImageNet) |
| Backend | Flask + Gunicorn |
| Frontend | HTML5 / CSS3 / Vanilla JS |
| Deployment | Hugging Face Spaces (Docker) |
| Training | Google Colab (T4 GPU) |

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**.
Not a certified medical device. Always consult a qualified physician.

---

## 👤 Author

**Atharva Barde**
B.Tech Computer Science · K.J. Somaiya Institute of Technology
[LinkedIn](https://linkedin.com/in/atharva-barde-b996b7384)
