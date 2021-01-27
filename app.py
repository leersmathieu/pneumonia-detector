import cv2 as cv
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import PIL
import base64
import io

st.set_page_config(
    page_title="Pneumonia Detection",
    layout="wide"
)

# variables #
preview_width = 500
input_size_1 = 128
input_size_2 = 128
# # # # # # #

row1_1, row1_2 = st.beta_columns((1, 1))

st.title("Pneumonia Detector")

# loading model
model = load_model("model/pneumonia_A91_R96_AUC94_128x128.h5")

uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
if uploaded_file is not None:

    # decode image with CV2
    opencv_image = cv.imdecode(np.frombuffer(uploaded_file.read(), dtype=np.uint8), 1)

    # image resizing
    resized = cv.resize(opencv_image, (input_size_1, input_size_2))
    normalized = resized / 255
    with row1_1:
        st.title("what's the computer see")
        st.image(resized, width=preview_width)

    # reshaping
    reshaped = np.expand_dims(normalized, axis=0)

    # prediction
    result = model.predict(reshaped)
    if result < 0.5:
        with row1_2:
            res = 100 - (result[0][0] * 100)
            cv.putText(opencv_image, "Normal", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
            cv.putText(opencv_image, str(res), (100, 200), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    else:
        with row1_2:
            res = (result[0][0] * 100)
            cv.putText(opencv_image, "Pneumonia", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
            cv.putText(opencv_image, str(res), (100, 200), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

    # display base image with the prediction integrated
    with row1_2:
        st.title("Prediction")
        st.image(opencv_image, width=preview_width)

        # download image with prediction
        download = st.button('Download JPEG File')
        if download:
            dl = PIL.Image.fromarray(opencv_image)
            buffered = io.BytesIO()
            dl.save(buffered, format="JPEG")
            dl_str = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:file/jpg;base64,{dl_str}" download="result.jpg">Click here to download your file</a>'
            st.markdown(href, unsafe_allow_html=True)
