import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

st.set_page_config(page_title="Classificateur d'images VGG16", page_icon="🌸", layout="centered")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model.h5",
        custom_objects={"preprocess_input": preprocess_input}
    )
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

IMG_SIZE = (224, 224)

def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    return array

st.title("🌸 Classificateur d'images (VGG16 - Transfer Learning)")
st.write("Téléversez une image pour obtenir une prédiction en temps réel.")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléversée", use_container_width=True)
    with st.spinner("Analyse en cours..."):
        processed = preprocess_image(image)
        predictions = model.predict(processed)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100
    st.success(f"**Classe prédite : {predicted_class}** ({confidence:.2f}% de confiance)")
    st.subheader("Scores de probabilité par classe")
    proba_dict = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    st.bar_chart(proba_dict)
else:
    st.info("En attente d'une image...")
