import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset"))
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model.h5")
THRESHOLD_PATH = os.path.join(BASE_DIR, "deepfake_threshold.txt")


def collect_paths_and_labels():
    paths = []
    labels = []

    for name in os.listdir(REAL_DIR):
        p = os.path.join(REAL_DIR, name)
        if os.path.isfile(p):
            paths.append(p)
            labels.append(0)  # REAL

    for name in os.listdir(FAKE_DIR):
        p = os.path.join(FAKE_DIR, name)
        if os.path.isfile(p):
            paths.append(p)
            labels.append(1)  # FAKE

    return np.array(paths), np.array(labels, dtype=np.int32)


def load_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype("float32") / 255.0


def build_dataset(paths, labels, training=False):
    images = []
    clean_labels = []

    for p, y in zip(paths, labels):
        img = load_and_resize(p)
        if img is not None:
            images.append(img)
            clean_labels.append(y)

    x = np.array(images, dtype=np.float32)
    y = np.array(clean_labels, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, x, y


def build_model():
    data_aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="data_augmentation",
    )

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_aug(inputs)
    # MobileNetV2 expected range is [-1, 1]; app inference provides [0, 1].
    x = tf.keras.layers.Lambda(lambda t: t * 2.0 - 1.0, name="mobilenet_scale")(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def best_threshold(y_true, y_prob):
    best_t = 0.5
    best_f1 = -1.0

    for t in np.arange(0.25, 0.76, 0.01):
        y_pred = (y_prob >= t).astype(np.int32)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


def main():
    paths, labels = collect_paths_and_labels()

    x_train_paths, x_temp_paths, y_train, y_temp = train_test_split(
        paths,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=SEED,
    )

    x_val_paths, x_test_paths, y_val, y_test = train_test_split(
        x_temp_paths,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=SEED,
    )

    train_ds, _, _ = build_dataset(x_train_paths, y_train, training=True)
    val_ds, _, _ = build_dataset(x_val_paths, y_val)
    test_ds, _, test_labels = build_dataset(x_test_paths, y_test)

    model, base = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=4, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

    # Fine-tune top layers of backbone.
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks)

    # Tune threshold on validation split (FAKE is positive class = 1).
    val_probs = model.predict(val_ds, verbose=0).reshape(-1)
    tuned_threshold, tuned_f1 = best_threshold(y_val, val_probs)

    test_probs = model.predict(test_ds, verbose=0).reshape(-1)
    y_pred_default = (test_probs >= 0.5).astype(np.int32)
    y_pred_tuned = (test_probs >= tuned_threshold).astype(np.int32)

    test_acc_default = np.mean(y_pred_default == test_labels)
    test_acc_tuned = np.mean(y_pred_tuned == test_labels)

    model.save(MODEL_PATH)
    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        f.write(f"{tuned_threshold:.4f}\n")

    print(f"Model saved: {MODEL_PATH}")
    print(f"Threshold saved: {THRESHOLD_PATH}")
    print(f"Validation best F1 threshold: {tuned_threshold:.4f} (F1={tuned_f1:.4f})")
    print(f"Test accuracy @0.50: {test_acc_default:.4f}")
    print(f"Test accuracy @{tuned_threshold:.2f}: {test_acc_tuned:.4f}")


if __name__ == "__main__":
    main()
