import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.models import load_model
from PIL import Image
import io
import pandas as pd

st.title("Deteksi Minuman")

# Load model
model = load_model('model.h5')
df_grading = pd.read_csv('./Dataset_Grading/Data Minuman - Data Minuman.csv')

# Load labels from JSON
with open('class_indices.json', 'r') as f:
    labels = json.load(f)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((150, 150))
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_nutrifacts(drink_name):
    drink = df_grading[df_grading['Produk'] == drink_name]
    drink = drink[['Gula/Sajian(g)', 'Gula/100ml(g)', 'Grade']].iloc[0]
    return drink

# Capture image from camera
picture = st.camera_input("Silakan Ambil Gambar")

if picture:
    # Display the captured image
    st.image(picture)

    # Preprocess the image
    img_bytes = picture.getvalue()
    img = preprocess_image(img_bytes)

    # Perform inference
    classes = model.predict(img)
    predicted_class_index = np.argmax(classes, axis=1)[0]
    predicted_class_label = labels[str(predicted_class_index)]
    
    # Display the prediction
    st.write(f'Predicted: {predicted_class_label}')
    nutrifacts = get_nutrifacts(predicted_class_label)
    st.write(f'Gula/100ml(g): {nutrifacts["Gula/100ml(g)"]}')
    st.write(f'Grade: {nutrifacts["Grade"]}')
    #input if u buy 1 bottle
    ml = int(st.number_input('Jumlah ukuran kontainer (ml)', min_value=1, max_value=2000, value=1, step=1))
    if ml > 0:
        gula = (ml * (nutrifacts["Gula/100ml(g)"]))/100
        st.write(f'Gula dalam minuman: {gula} gram')
