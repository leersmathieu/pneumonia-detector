import streamlit as st
import io
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model


def conversion(img):
    x = cv2.imdecode(np.frombuffer(img.read(), dtype=np.uint8), 1)
    x = cv2.resize(x, (64, 64))
    x = x / 255.
    print(x.dtype)
    img_data = np.expand_dims(x, axis=0)
    return img_data


def predict(img_data):
    model = load_model('model/pneumonia_A87_R92_AUC94.h5')
    classes = model.predict(img_data)
    print(classes)
    result = np.round(classes[0][0])
    print(result)
    return result


uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
if uploaded_file is not None:
    st.write('You selected `%s`' % uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Analyzing...")
    res = conversion(uploaded_file)
    pred = predict(res)
    st.write(pred)
    if pred < 0.5:
        st.write("Result is Normal")

    else:
        st.write("Person is infected By PNEUMONIA")

