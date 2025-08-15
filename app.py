from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
from flask_cors import CORS

CORS(app)

# Carregar modelo SavedModel como camada TFSMLayer (inference only)
model = tf.keras.Sequential([
    tf.keras.layers.TFSMLayer("model.savedmodel", call_endpoint="serving_default")
])

# Carregar labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Função para pré-processar igual ao treino no Teachable Machine
def preprocess_image(image):
    img = image.resize((224, 224))  # Teachable Machine usa 224x224
    img_array = np.asarray(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1  # normalização específica do TM
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    img_array = preprocess_image(image)

    prediction = model(img_array)  # note que agora chamamos o modelo como layer
    predicted_index = int(np.argmax(prediction))
    predicted_label = labels[predicted_index]
    confidence = float(prediction[0][predicted_index])

    return jsonify({
        "class": predicted_label,
        "confidence": confidence
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
