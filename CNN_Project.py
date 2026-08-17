from pathlib import Path
import pickle

import numpy as np
import streamlit as st
from PIL import Image


CLASS_NAMES = ["surprise", "sad", "neutral", "happy", "fear", "angry", "disgust"]
DEFAULT_MODEL_PATH = Path(r"C:\Users\shami\cnn_model.pkl")
IMAGE_SIZE = (48, 48)


st.set_page_config(
    page_title="Facial Emotion CNN",
    page_icon=":)",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1120px;
    }
    .prediction {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.25rem 0 0.75rem;
    }
    .muted {
        color: #64748b;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_from_path(model_path: str):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix.lower() == ".keras":
        import tensorflow as tf

        return tf.keras.models.load_model(path)

    with path.open("rb") as f:
        return pickle.load(f)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L")
    image = image.resize(IMAGE_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_emotion(model, image: Image.Image):
    arr = preprocess_image(image)
    probs = model.predict(arr, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    return predicted_idx, probs


def show_confidence_bars(probs):
    order = np.argsort(probs)[::-1]
    for idx in order:
        label = CLASS_NAMES[idx]
        value = float(probs[idx])
        st.write(f"**{label}** `{value:.2%}`")
        st.progress(value)


st.title("Facial Emotion Recognition")
st.caption("Upload a face image and classify it with your trained CNN model.")

with st.sidebar:
    st.header("Model")
    model_path = DEFAULT_MODEL_PATH

    st.header("Classes")
    st.write(", ".join(CLASS_NAMES))


model = None
model_error = None

try:
    model = load_model_from_path(model_path)
except Exception as exc:
    model_error = exc


else:
    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "bmp"],
        )

        if uploaded_file is None:
            st.info("Upload a face image to get a prediction.")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_container_width=True)

    with right:
        if uploaded_file is not None:
            try:
                predicted_idx, probs = predict_emotion(model, image)
                predicted_label = CLASS_NAMES[predicted_idx]
                confidence = float(probs[predicted_idx])

                st.subheader("Prediction")
                st.markdown(
                    f'<div class="prediction">{predicted_label}</div>',
                    unsafe_allow_html=True,
                )
                st.metric("Confidence", f"{confidence:.2%}")

                st.subheader("Class Confidence")
                show_confidence_bars(probs)
            except Exception as exc:
                st.error("Prediction failed.")
                st.code(str(exc))
