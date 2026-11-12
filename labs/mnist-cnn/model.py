"""CNN and data preparation for the MNIST lab."""

from pathlib import Path

import numpy as np


LAB_DIR = Path(__file__).resolve().parent
PROVIDED_MODEL = LAB_DIR / "best_model.keras"
TRAINED_MODEL = LAB_DIR / "artifacts" / "best_model.keras"


def prepare_images(images):
    return np.asarray(images, dtype=np.float32)[..., np.newaxis] / 255.0


def load_splits(seed=42):
    from tensorflow.keras.datasets import mnist

    (images, labels), (test_images, test_labels) = mnist.load_data()
    order = np.random.default_rng(seed).permutation(len(images))
    val, train = np.split(order, [len(images) // 10])
    return (
        (prepare_images(images[train]), labels[train]),
        (prepare_images(images[val]), labels[val]),
        (prepare_images(test_images), test_labels),
    )


def define_model():
    from tensorflow import keras

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu", kernel_initializer="he_uniform"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="relu", kernel_initializer="he_uniform"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(train, validation, epochs=5, seed=42):
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    TRAINED_MODEL.parent.mkdir(exist_ok=True)
    model = define_model()
    checkpoint = keras.callbacks.ModelCheckpoint(
        TRAINED_MODEL, monitor="val_loss", mode="min", save_best_only=True,
    )
    history = model.fit(
        *train, validation_data=validation, epochs=epochs, batch_size=128,
        callbacks=[checkpoint], verbose=2,
    )
    return keras.models.load_model(TRAINED_MODEL), history


def load_model(model_path=None):
    if model_path is None:
        model_path = TRAINED_MODEL if TRAINED_MODEL.is_file() else PROVIDED_MODEL
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError("Chạy train_model.py hoặc đặt best_model.keras vào thư mục lab.")

    from tensorflow import keras

    return keras.models.load_model(model_path, compile=False)
