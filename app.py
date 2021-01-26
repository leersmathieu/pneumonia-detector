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

######################

row1_1, row1_2 = st.beta_columns((2, 4))


def conversion(img):
    x = cv2.imdecode(np.frombuffer(img.read(), dtype=np.uint8), 1)
    x = cv2.resize(x, (input_size_1, input_size_2))
    st.write("The computer see...")
    st.image(x, width=500)
    x = x / 255.
    img_data = np.expand_dims(x, axis=0)
    return img_data


def predict(img_data):
    model = load_model('model/pneumonia_A88_R94_AUC95_128x128.h5')
    classes = model.predict(img_data)
    print(classes)
    st.write(classes[0][0])
    result = np.round(classes[0][0])
    print(result)
    return result


uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
if uploaded_file is not None:
    st.write('You selected `%s`' % uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", width=500)
    st.write("")
    st.write("Analyzing...")
    res = conversion(uploaded_file)
    pred = predict(res)
    if pred < 0.5:
        st.write("Result is Normal")

    else:
        st.write("Person is infected By PNEUMONIA")

