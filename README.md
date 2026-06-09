# Emotion Recognition — SPbSTU ML Final Project

Real-time facial emotion detection via webcam. Flask backend + browser frontend, model trained on H100 GPU.

## Classes

| Emotion | Color |
|---|---|
| 😠 Angry | Red |
| 😊 Happy | Green |
| 😢 Sad | Blue |
| 😲 Surprised | Orange |
| 😐 Neutral | Gray |

## Results

| Metric | Value |
|---|---|
| Architecture | ResNet50V2 (fine-tuned) |
| Dataset | FER-2013 + RAF-DB (~51k images) |
| Val accuracy | **78.9%** |
| GPU | NVIDIA H100 80GB (RunPod) |
| Training time | ~18 min |

## Quick start

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5001
```

Allow camera access in the browser. The model updates the emotion label once per second.

If no face is detected → "No face detected" is shown.

## How it works

1. Browser captures webcam frame every 1s via `getUserMedia()`
2. Sends JPEG to `POST /predict`
3. Server runs Haar Cascade face detection → crops face → ResNet50V2 inference
4. Returns emotion + confidence + bounding box
5. Browser draws colored bbox + emoji overlay on the video

## Re-training

To retrain on RunPod:

```bash
# 1. Prepare dataset (FER-2013 + RAF-DB)
python prepare_data.py

# 2. Train
LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH \
python train.py
```

Requires `~/.kaggle/kaggle.json` with valid Kaggle API credentials.
