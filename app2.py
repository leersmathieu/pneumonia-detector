import cv2 as cv
import os
import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

# important variables

input_size_1 = 128
input_size_2 = 128
dim = 3

######################

model = load_model("model/pneumonia_A88_R94_AUC95_128x128.h5")


uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
if uploaded_file is not None:
    category = {0: 'Pneumonia', 1: 'Normal'}
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv.imdecode(file_bytes, 1)
    grayImage = cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
    resized = cv.resize(grayImage, (input_size_1, input_size_2))
    normalized = resized / 255
    reshaped = np.reshape(normalized, (1, input_size_1, input_size_2, 1))
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    prob = np.max(result, axis=1)[0]
    prob = round(prob, 2) * 100
    # image[:50,:]=[0,255,0]
    cv.putText(opencv_image, str(category[label]), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    cv.putText(opencv_image, str(prob), (100, 200), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    st.image(opencv_image)