import streamlit as st
import io
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# important variables

input_size_1 = 128
input_size_2 = 128
dim = 3
preview_width = 500

######################

def conversion(img):
    x = cv2.imdecode(np.frombuffer(img.read(), dtype=np.uint8), 1)
    x = cv2.resize(x, (input_size_1, input_size_2))
    st.write("The computer see...")
    st.image(x, width=preview_width)
    x = x / 255.
    img_data = np.expand_dims(x, axis=0)
    return img_data


def predict(img_data):
    model = load_model('model/pneumonia_A88_R94_AUC95_128x128.h5')
    classes = model.predict(img_data)
    result = np.round(classes[0][0])
    percent = round(classes[0][0], 2)
    return result, percent


uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
if uploaded_file is not None:
    st.write('You selected `%s`' % uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", width=preview_width)
    st.write("")
    st.write("Analyzing...")
    res = conversion(uploaded_file)
    pred, percent = predict(res)

    if pred < 0.5:
        st.write(100 - (percent * 100))
        st.write("Result is Normal")
    else:
        st.write(percent * 100)
        st.write("Person is infected By PNEUMONIA")
