import os, glob, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from PIL import Image

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("GPUs:", gpus)

IMG_SIZE   = 128
BATCH_SIZE = 512
CLASSES    = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
DATA_DIR   = "/workspace/data_merged/train"
VAL_DIR    = "/workspace/data_merged/test"
OUT_DIR    = "/workspace"
BEST_W     = f"{OUT_DIR}/emotion_model.h5"

# Phase 1: train head only (base frozen)
EPOCHS_HEAD     = 10
LR_HEAD         = 1e-3

# Phase 2: fine-tune entire network
EPOCHS_FINETUNE = 60
LR_FINETUNE     = 1e-5


def load_split(split_dir):
    images, labels = [], []
    for idx, cls in enumerate(CLASSES):
        paths = sorted(
            glob.glob(f"{split_dir}/{cls}/*.jpg") +
            glob.glob(f"{split_dir}/{cls}/*.png")
        )
        print(f"  {cls}: {len(paths)}", flush=True)
        for p in paths:
            try:
                img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(idx)
            except Exception:
                pass
    return np.array(images), np.array(labels)


def build_model():
    base = keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None,
    )
    base.trainable = False

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Augmentation (applied only during training by Keras)
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomRotation(0.10)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.RandomBrightness(0.10)(x)
    x = layers.RandomContrast(0.10)(x)

    # ResNet50V2 expects inputs in [-1, 1]
    x = keras.applications.resnet_v2.preprocess_input(x * 255.0)

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASSES), activation="softmax", dtype="float32")(x)

    model = keras.Model(inp, out)
    return model, base


print("Loading train...", flush=True)
X_train, y_train = load_split(DATA_DIR)
print(f"Train total: {X_train.shape}")

print("Loading val...", flush=True)
X_val, y_val = load_split(VAL_DIR)
print(f"Val total:   {X_val.shape}")

y_train_cat = tf.keras.utils.to_categorical(y_train, len(CLASSES))
y_val_cat   = tf.keras.utils.to_categorical(y_val,   len(CLASSES))

cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))
print("Class weights:", {k: round(float(v), 2) for k, v in cw_dict.items()})

model, base_model = build_model()
model.summary()

# ── Phase 1: train head, base frozen ────────────────────────────────────────
print("\n=== Phase 1: training head (base frozen) ===")
model.compile(
    optimizer=keras.optimizers.Adam(LR_HEAD),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history1 = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE, epochs=EPOCHS_HEAD,
    validation_data=(X_val, y_val_cat),
    class_weight=cw_dict,
    verbose=1,
)

# ── Phase 2: unfreeze all, fine-tune ────────────────────────────────────────
print("\n=== Phase 2: fine-tuning entire network ===")
base_model.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(LR_FINETUNE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                                      min_lr=1e-7, verbose=1),
    keras.callbacks.ModelCheckpoint(BEST_W, monitor="val_accuracy",
                                    save_best_only=True, verbose=1),
]

history2 = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE, epochs=EPOCHS_FINETUNE,
    validation_data=(X_val, y_val_cat),
    callbacks=callbacks,
    class_weight=cw_dict,
    verbose=1,
)

model.load_weights(BEST_W)
print("Loaded best weights from checkpoint.")

# Merge histories
full_history = {}
for k in history1.history:
    full_history[k] = history1.history[k] + history2.history.get(k, [])

with open(f"{OUT_DIR}/training_history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in full_history.items()}, f, indent=2)

loss, acc = model.evaluate(X_val, y_val_cat, verbose=0)
print(f"\nVal accuracy: {acc:.4f}  loss: {loss:.4f}")

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
print(classification_report(y_val, y_pred, target_names=CLASSES,
                             labels=list(range(len(CLASSES)))))

meta = {
    "val_accuracy": float(acc),
    "val_loss": float(loss),
    "img_size": IMG_SIZE,
    "classes": CLASSES,
    "epochs_phase1": EPOCHS_HEAD,
    "epochs_phase2": len(history2.history["loss"]),
    "architecture": "ResNet50V2 fine-tuned",
}
with open(f"{OUT_DIR}/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Done.")
