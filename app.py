import os, json
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH  = os.path.join(BASE_DIR, "model_meta.json")
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")

with open(META_PATH) as f:
    meta = json.load(f)
CLASSES = meta["classes"]
IMG_SIZE = meta["img_size"]

model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print(f"Model loaded. Classes: {CLASSES}")

app = Flask(__name__)


@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "index.html"))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.data
    if not data:
        return jsonify({"emotion": None})

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"emotion": None})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return jsonify({"emotion": None})

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    roi = frame[y:y + h, x:x + w]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = roi.astype("float32") / 255.0
    img = np.expand_dims(img, 0)

    preds = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])

    return jsonify({
        "emotion": CLASSES[idx].capitalize(),
        "confidence": round(confidence, 3),
        "bbox": [int(x), int(y), int(w), int(h)],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
