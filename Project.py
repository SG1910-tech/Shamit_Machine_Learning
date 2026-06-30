import numpy as np
import streamlit as st
from PIL import Image
import pickle

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = expZ / np.sum(expZ, axis=0, keepdims=True)
    return A

def relu(Z):
    return np.maximum(0, Z)

def linear_forward(A_prev, W, b):
    return np.dot(W, A_prev) + b

def model_forward(X, parameters):
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        Z = linear_forward(A, parameters['W' + str(l)], parameters['b' + str(l)])
        A = relu(Z)
    ZL = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
    AL = softmax(ZL)
    return AL

with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

parameters = model_data["parameters"]
class_names = model_data["class_names"]
image_size = model_data["image_size"]

st.title("Image Classification App")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type = ["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption = "Uploaded Image", use_container_width = True)
    image_resized = image.resize(image_size)
    image_array = np.array(image_resized)
    X = image_array.reshape(1, -1)
    X = X/255
    X = X.T

    probabilities = model_forward(X ,parameters)

    predicted_index = np.argmax(probabilities, axis = 0)[0]
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index, 0]

    st.subheader("Prediction")
    st.write("Class:", predicted_class)
    st.write("Confidence:", (confidence * 100), "%")

    st.subheader("Class Probabilities")

    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {probabilities[i, 0] * 100:.2f}%")
        st.progress(float(probabilities[i, 0]))