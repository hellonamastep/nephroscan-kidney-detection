"""
╔══════════════════════════════════════════════════════════════╗
║   NephroScan — Kidney Stone Detection                        ║
║   Flask Backend · Hugging Face Spaces Ready                  ║
║   Author: Atharva Barde                                      ║
╚══════════════════════════════════════════════════════════════╝

Local run  : python app.py  →  http://localhost:5000
Production : gunicorn --bind 0.0.0.0:7860 --timeout 120 app:app
"""

import os
import json
import numpy as np
from PIL import Image
import io
import base64

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("model", "nephroscan_final.keras")
INDICES_PATH = os.path.join("model", "class_indices.json")
IMG_SIZE     = (224, 224)

os.makedirs("model",   exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# AUTO-DOWNLOAD MODEL FROM HUGGING FACE HUB
# (only runs if model files are missing — e.g. first deploy)
# ─────────────────────────────────────────────────────────────
HF_REPO_ID = os.environ.get("HF_REPO_ID", "atharvabarde/nephroscan")

def download_model_if_needed():
    """Download model weights from HF Hub if not present locally."""
    files_needed = [
        ("nephroscan_final.keras", MODEL_PATH),
        ("class_indices.json",     INDICES_PATH),
    ]
    missing = [f for f, p in files_needed if not os.path.exists(p)]

    if not missing:
        print("✅ Model files found locally.")
        return

    print(f"⬇️  Downloading {missing} from Hugging Face Hub...")
    try:
        from huggingface_hub import hf_hub_download
        for filename, local_path in files_needed:
            if not os.path.exists(local_path):
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=filename,
                    local_dir="model",
                    local_dir_use_symlinks=False,
                )
                print(f"   ✅ Downloaded: {filename}")
    except Exception as e:
        print(f"   ⚠️  Could not download from Hub: {e}")
        print("      Place model files manually in the model/ folder.")

download_model_if_needed()

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
print("\n🔄 Loading TensorFlow model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}.\n"
        "  → Run training first (NephroScan_Colab.ipynb)\n"
        "  → Then place nephroscan_final.keras in the model/ folder."
    )

model = tf.keras.models.load_model(MODEL_PATH)

with open(INDICES_PATH) as f:
    class_indices = json.load(f)

idx_to_class = {int(v): k for k, v in class_indices.items()}
print(f"✅ Model loaded  |  Classes: {list(idx_to_class.values())}")
print(f"   Parameters   : {model.count_params():,}")
print(f"   Input shape  : {model.input_shape}\n")

# ─────────────────────────────────────────────────────────────
# CLINICAL DESCRIPTIONS
# ─────────────────────────────────────────────────────────────
DESCRIPTIONS = {
    "Stone": {
        "color"         : "#ef4444",
        "icon"          : "⚠️",
        "severity"      : "DETECTED",
        "severity_color": "#ef4444",
        "description"   : (
            "Kidney stones (nephrolithiasis) are hard mineral and salt deposits "
            "that form inside the kidneys. They can cause severe flank pain, "
            "nausea, vomiting, and blood in the urine. "
            "Early medical attention is strongly recommended."
        ),
        "recommendation": (
            "Consult a urologist promptly. Treatment depends on stone size — "
            "small stones may pass with hydration, while larger ones may require "
            "lithotripsy or surgical intervention."
        ),
    },
    "Cyst": {
        "color"         : "#f59e0b",
        "icon"          : "🔍",
        "severity"      : "MONITOR",
        "severity_color": "#f59e0b",
        "description"   : (
            "A renal cyst is a round, fluid-filled sac that grows on or within "
            "the kidney. Simple cysts are typically benign and asymptomatic, "
            "but complex cysts may require further evaluation."
        ),
        "recommendation": (
            "Schedule a follow-up with a nephrologist. Most simple cysts "
            "are benign and only require periodic imaging surveillance. "
            "Complex cysts may need biopsy or drainage."
        ),
    },
    "Tumor": {
        "color"         : "#dc2626",
        "icon"          : "🚨",
        "severity"      : "URGENT",
        "severity_color": "#dc2626",
        "description"   : (
            "A renal mass or tumor has been identified in the CT scan. "
            "This finding requires urgent specialist evaluation. "
            "Renal cell carcinoma (RCC) is the most common type, but "
            "early detection significantly improves treatment outcomes."
        ),
        "recommendation": (
            "Seek immediate evaluation by a urologist or oncologist. "
            "Further imaging (MRI/PET scan) and possible biopsy will be required "
            "to determine the nature and staging of the mass."
        ),
    },
    "Normal": {
        "color"         : "#22c55e",
        "icon"          : "✅",
        "severity"      : "NORMAL",
        "severity_color": "#22c55e",
        "description"   : (
            "The CT scan appears within normal limits. No stones, cysts, "
            "or masses are detectable. Kidney morphology, size, and density "
            "are all within the expected range for a healthy renal system."
        ),
        "recommendation": (
            "No immediate action is required. Maintain a healthy lifestyle "
            "with adequate hydration. Continue routine annual health check-ups "
            "as recommended by your physician."
        ),
    },
}

# ─────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tiff"}


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(image_bytes: bytes):
    """Decode bytes → PIL Image → normalized numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), img


# ── Routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Validate file presence
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please attach an image."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Please upload PNG, JPG, JPEG, BMP, or WEBP."
        }), 400

    try:
        image_bytes = file.read()

        # Preprocess
        input_array, pil_img = preprocess_image(image_bytes)

        # Inference
        raw_preds = model.predict(input_array, verbose=0)[0]   # shape (4,)
        pred_idx  = int(np.argmax(raw_preds))
        pred_class = idx_to_class[pred_idx]
        confidence = float(raw_preds[pred_idx]) * 100

        # All class probabilities (sorted descending)
        probs_dict = {
            idx_to_class[i]: round(float(raw_preds[i]) * 100, 2)
            for i in range(len(raw_preds))
        }
        probs_sorted = dict(
            sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        )

        # Thumbnail (base64) for response
        buf = io.BytesIO()
        pil_img.thumbnail((320, 320), Image.LANCZOS)
        pil_img.save(buf, format="JPEG", quality=88)
        thumbnail_b64 = base64.b64encode(buf.getvalue()).decode()

        info = DESCRIPTIONS[pred_class]

        return jsonify({
            "success"       : True,
            "prediction"    : pred_class,
            "confidence"    : round(confidence, 2),
            "probabilities" : probs_sorted,
            "severity"      : info["severity"],
            "severity_color": info["severity_color"],
            "color"         : info["color"],
            "icon"          : info["icon"],
            "description"   : info["description"],
            "recommendation": info["recommendation"],
            "thumbnail"     : f"data:image/jpeg;base64,{thumbnail_b64}",
        })

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({
        "status" : "ok",
        "model"  : "NephroScan v1.0",
        "classes": list(idx_to_class.values()),
    })


@app.route("/classes")
def get_classes():
    return jsonify({"classes": CLASSES_INFO})


CLASSES_INFO = {
    cls: {
        "color"   : DESCRIPTIONS[cls]["color"],
        "icon"    : DESCRIPTIONS[cls]["icon"],
        "severity": DESCRIPTIONS[cls]["severity"],
    }
    for cls in DESCRIPTIONS
}

# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    print(f"\n🚀 NephroScan server starting...")
    print(f"   URL   : http://localhost:{port}")
    print(f"   Debug : {debug}\n")
    app.run(debug=debug, host="0.0.0.0", port=port)
