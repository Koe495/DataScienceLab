"""Local Flask demo for drawing and recognizing a single handwritten digit."""

import base64
import binascii
import io
from functools import lru_cache

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

from model import load_model


def preprocess_image(data_url):
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("Dữ liệu ảnh phải là PNG.")

    try:
        raw = base64.b64decode(data_url[len(prefix):], validate=True)
        with Image.open(io.BytesIO(raw), formats=["PNG"]) as source:
            if source.width > 1024 or source.height > 1024:
                raise ValueError("Kích thước ảnh tối đa là 1024 × 1024.")
            rgba = source.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
            img = Image.alpha_composite(background, rgba).convert("L")
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Ảnh PNG không hợp lệ.") from exc

    # The canvas uses a black background with white strokes, as in MNIST.
    if np.asarray(img).mean() > 127:
        img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox is None:
        raise ValueError("Hãy vẽ một chữ số trước khi dự đoán.")

    digit = img.crop(bbox)
    scale = 20 / max(digit.size)
    width, height = (max(1, round(side * scale)) for side in digit.size)
    digit = digit.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28))
    canvas.paste(digit, ((28 - width) // 2, (28 - height) // 2))
    return np.asarray(canvas, dtype=np.float32)[None, ..., None] / 255.0


def create_app(model_loader=load_model):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024
    get_model = lru_cache(maxsize=1)(model_loader)

    @app.get("/")
    def index():
        return render_template("digits.html")

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "image" not in data:
            return jsonify(error="Thiếu dữ liệu ảnh."), 400
        try:
            inputs = preprocess_image(data["image"])
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        try:
            model = get_model()
        except FileNotFoundError as exc:
            return jsonify(error=str(exc)), 503
        probabilities = model.predict(inputs, verbose=0)[0]
        digit = int(np.argmax(probabilities))
        return jsonify(
            prediction=digit,
            confidence=round(float(probabilities[digit]) * 100, 2),
            probabilities=[round(float(score) * 100, 2) for score in probabilities],
        )

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="Dữ liệu ảnh vượt quá dung lượng cho phép."), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
