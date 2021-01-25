import streamlit as st
import PIL
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tempfile import NamedTemporaryFile


def conversion(img):
    x = img.convert("RGB")
    x = x.resize((224, 224), Image.NEAREST)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    return img_data


def predict(img_data):
    model = load_model('chest_xray_pneumonia.h5')
    classes = model.predict(img_data)
    result = int(classes[0][0])
    return result

def load_image(img_file):
    img = Image.open(img_file)
    return img

def main():
    uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
    if uploaded_file is not None:
        s = PIL.Image.open(uploaded_file)
        st.image(s, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Analyzing...")
        res = conversion(s)
        pred = predict(res)
        st.write(pred)
        if pred == 0:
            st.write("Person is Affected By PNEUMONIA")
        else:
            st.write("Result is Normal")


if __name__ == "__main__":
    main()
