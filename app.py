from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Crear el modelo MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation="softmax")
])

# Cargar los pesos entrenados
model.load_weights("modelo_basura.weights.h5")

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    resultado = None
    confianza = None
    imagen = None

    if request.method == "POST":

        archivo = request.files["imagen"]

        if archivo:

            ruta = os.path.join(UPLOAD_FOLDER, archivo.filename)
            archivo.save(ruta)

            imagen_modelo = load_img(
                ruta,
                target_size=(224, 224)
            )

            imagen_array = img_to_array(imagen_modelo)
            imagen_array = np.expand_dims(imagen_array, axis=0)

            prediccion = model.predict(imagen_array)

            indice = np.argmax(prediccion[0])

            resultado = class_names[indice].upper()
            confianza = round(
                float(prediccion[0][indice]) * 100,
                2
            )

            imagen = "/" + ruta

    return render_template(
        "index.html",
        resultado=resultado,
        confianza=confianza,
        imagen=imagen
    )


if __name__ == "__main__":
    app.run(debug=True)