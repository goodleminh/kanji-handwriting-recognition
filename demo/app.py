"""
==========================================================
  Kanji Handwriting Recognition - Flask Backend
==========================================================
  Load 2 models (CNN & ResNet18) and serve prediction API
  for handwritten Kanji recognition from canvas.
==========================================================
"""
 
import os
import sys
import json
import base64
import io
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Fix Unicode output on Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==================== CONFIG ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
LABEL_DIR = os.path.join(BASE_DIR, 'label')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

IMG_SIZE = 64  # Input image size for both models

# Dataset in training: black background, white strokes.
# Canvas often produces black strokes on white background -> invert.
INVERT_CANVAS = True

# Reduce TF startup logs (must be set before importing tensorflow)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# Silence oneDNN info spam on Windows; does not affect correctness.
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# ==================== INIT APP ====================
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# ==================== GLOBAL VARS ====================
cnn_model = None
resnet_model = None
cnn_labels = None
resnet_labels = None


def load_models():
    """Load both models and label files on server startup."""
    global cnn_model, resnet_model, cnn_labels, resnet_labels

    # Import tensorflow here to catch import errors
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        print("[WARN] TensorFlow is not installed!")
        print("       Run: pip install tensorflow")
        return

    # --- Keras compatibility patch ---
    # Models saved on Colab (newer Keras) may include 'quantization_config'
    # in layer configs, which older local Keras versions don't recognize.
    # Patch Dense to drop this key to keep loading legacy .h5 models working.
    try:
        # Patch BOTH standalone keras and tf.keras Dense (they may differ depending on install)
        try:
            from keras.layers import Dense as _KerasDense
        except Exception:
            _KerasDense = None
        try:
            _TfKerasDense = tf.keras.layers.Dense
        except Exception:
            _TfKerasDense = None

        def _patch_dense(DenseCls):
            if DenseCls is None:
                return
            orig_init = DenseCls.__init__

            def patched_init(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                return orig_init(self, *args, **kwargs)

            DenseCls.__init__ = patched_init

        _patch_dense(_KerasDense)
        _patch_dense(_TfKerasDense)

        print("[OK] Keras compatibility patch applied")
    except Exception as e:
        print(f"[WARN] Could not apply Keras patch: {e}")

    # Helper: try normal load, then fallback using safe_mode=False
    def _safe_load_model(path: str):
        try:
            return tf.keras.models.load_model(path)
        except TypeError as e:
            # Some Keras3 deserialization errors can be bypassed with safe_mode=False
            try:
                return tf.keras.models.load_model(path, safe_mode=False)
            except Exception:
                raise e

    # --- Load CNN Model ---
    cnn_path = os.path.join(MODEL_DIR, 'model_cnn.h5')
    if os.path.exists(cnn_path):
        try:
            cnn_model = _safe_load_model(cnn_path)
            print(f"[OK] CNN model loaded: {cnn_path}")
            print(f"     Input shape: {cnn_model.input_shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load CNN model: {e}")
    else:
        print(f"[WARN] CNN model not found: {cnn_path}")

    # --- Load ResNet18 Model ---
    resnet_path = os.path.join(MODEL_DIR, 'model_resnet18.h5')
    if os.path.exists(resnet_path):
        try:
            resnet_model = _safe_load_model(resnet_path)
            print(f"[OK] ResNet18 model loaded: {resnet_path}")
            print(f"     Input shape: {resnet_model.input_shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load ResNet18 model: {e}")
    else:
        print(f"[WARN] ResNet18 model not found: {resnet_path}")

    # --- Load CNN Labels ---
    cnn_label_path = os.path.join(LABEL_DIR, 'labelscnn.json')
    if os.path.exists(cnn_label_path):
        try:
            with open(cnn_label_path, 'r', encoding='utf-8') as f:
                cnn_labels = json.load(f)
            num = cnn_labels.get('num_classes', 'N/A')
            print(f"[OK] CNN labels loaded: {cnn_label_path} ({num} classes)")
        except Exception as e:
            print(f"[ERROR] Failed to load CNN labels: {e}")
    else:
        print(f"[WARN] CNN labels not found: {cnn_label_path}")

    # --- Load ResNet18 Labels ---
    resnet_label_path = os.path.join(LABEL_DIR, 'labelsresnet18.json')
    if os.path.exists(resnet_label_path):
        try:
            with open(resnet_label_path, 'r', encoding='utf-8') as f:
                resnet_labels = json.load(f)
            num = resnet_labels.get('num_classes', 'N/A')
            print(f"[OK] ResNet18 labels loaded: {resnet_label_path} ({num} classes)")
        except Exception as e:
            print(f"[ERROR] Failed to load ResNet18 labels: {e}")
    else:
        print(f"[WARN] ResNet18 labels not found: {resnet_label_path}")


def decode_base64_image(base64_string):
    """
    Decode base64 image from canvas to PIL Image.
    Supports both 'data:image/png;base64,...' and raw base64.
    """
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return image


def _ensure_training_contrast(img_gray: Image.Image) -> Image.Image:
    """Ensure image matches training polarity: black background, white strokes."""
    if not INVERT_CANVAS:
        return img_gray
    # Canvas usually: white background (high mean), black strokes.
    arr = np.asarray(img_gray, dtype=np.uint8)
    if arr.mean() > 127:
        arr = 255 - arr
    return Image.fromarray(arr, mode='L')


def _expected_channels_from_labels(labels: dict, default: int) -> int:
    try:
        shape = labels.get('input_shape') if isinstance(labels, dict) else None
        if isinstance(shape, list) and len(shape) == 3:
            return int(shape[2])
    except Exception:
        pass
    return default


def preprocess_for_cnn(image):
    """
    Preprocess image for CNN model.
    Training expects: (64, 64, 1) grayscale, normalized /255.0,
    and polarity: black background, white strokes.
    """
    expected_c = _expected_channels_from_labels(cnn_labels or {}, default=1)
    if expected_c != 1:
        raise ValueError(f"CNN expected_channels mismatch: labels say {expected_c}, but pipeline is grayscale(1).")

    img_gray = image.convert('L')
    img_gray = _ensure_training_contrast(img_gray)
    img_resized = img_gray.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img_array


def preprocess_for_resnet(image):
    """
    Preprocess image for ResNet18 model.
    Training expects: (64, 64, 3) where grayscale is stacked to 3 channels,
    normalized /255.0, and polarity: black background, white strokes.
    """
    expected_c = _expected_channels_from_labels(resnet_labels or {}, default=3)
    if expected_c != 3:
        raise ValueError(f"ResNet expected_channels mismatch: labels say {expected_c}, but pipeline stacks to 3.")

    img_gray = image.convert('L')
    img_gray = _ensure_training_contrast(img_gray)
    img_resized = img_gray.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.stack([img_array] * 3, axis=-1)
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    return img_array


def get_predictions(model, img_array, labels, top_k=3):
    """
    Run inference and return top-k results.
    Returns dict: { top1, confidence, top3: [{label, prob}, ...] }
    """
    predictions = model.predict(img_array, verbose=0)
    probs = predictions[0]

    top_indices = np.argsort(probs)[::-1][:top_k]
    index_to_kanji = labels.get('index_to_kanji', {})

    top1_idx = top_indices[0]
    top1_label = index_to_kanji.get(str(top1_idx), f'class_{top1_idx}')
    top1_confidence = float(probs[top1_idx])

    top3 = []
    for idx in top_indices:
        label = index_to_kanji.get(str(idx), f'class_{idx}')
        prob = float(probs[idx])
        top3.append({
            'label': label,
            'prob': round(prob, 4)
        })

    return {
        'top1': top1_label,
        'confidence': round(top1_confidence, 4),
        'top3': top3
    }


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)."""
    return send_from_directory(FRONTEND_DIR, filename)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Kanji recognition API.

    Input (JSON):
      { "image": "data:image/png;base64,..." }

    Output (JSON):
      {
        "cnn": { "top1": "...", "confidence": 0.92, "top3": [...] },
        "resnet": { "top1": "...", "confidence": 0.89, "top3": [...] }
      }
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing image data. Send JSON with "image" key.'}), 400

        image = decode_base64_image(data['image'])
        result = {}

        # --- CNN Prediction ---
        if cnn_model is not None and cnn_labels is not None:
            try:
                img_cnn = preprocess_for_cnn(image)
                result['cnn'] = get_predictions(cnn_model, img_cnn, cnn_labels)
            except Exception as e:
                result['cnn'] = {
                    'error': f'CNN preprocessing/prediction failed: {str(e)}',
                    'top1': '?',
                    'confidence': 0,
                    'top3': []
                }
        else:
            result['cnn'] = {
                'error': 'CNN model or labels not loaded.',
                'top1': '?',
                'confidence': 0,
                'top3': []
            }

        # --- ResNet18 Prediction ---
        if resnet_model is not None and resnet_labels is not None:
            try:
                img_resnet = preprocess_for_resnet(image)
                result['resnet'] = get_predictions(resnet_model, img_resnet, resnet_labels)
            except Exception as e:
                result['resnet'] = {
                    'error': f'ResNet preprocessing/prediction failed: {str(e)}',
                    'top1': '?',
                    'confidence': 0,
                    'top3': []
                }
        else:
            result['resnet'] = {
                'error': 'ResNet18 model or labels not loaded.',
                'top1': '?',
                'confidence': 0,
                'top3': []
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/status', methods=['GET'])
def status():
    """Check server and model status."""
    return jsonify({
        'server': 'running',
        'cnn_model': 'loaded' if cnn_model is not None else 'not loaded',
        'resnet_model': 'loaded' if resnet_model is not None else 'not loaded',
        'cnn_labels': 'loaded' if cnn_labels is not None else 'not loaded',
        'resnet_labels': 'loaded' if resnet_labels is not None else 'not loaded',
    })


# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 60)
    print("  KANJI HANDWRITING RECOGNITION SERVER")
    print("=" * 60)

    load_models()

    print("=" * 60)
    print("  Server running at: http://localhost:5000")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False)
