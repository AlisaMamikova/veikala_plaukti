import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="CrowdHuman — YOLOv8 Counting", layout="wide")
st.title("Crowd Counting with YOLOv8 (CrowdHuman)")

weights = st.text_input("Weights (.pt)", value="runs/detect/train2/weights/best.pt")
device = st.selectbox("Device", options=["auto","cuda","mps","cpu"], index=0)
conf = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

@st.cache_resource
def load_model(weights_path, device_choice):
    d = None if device_choice == "auto" else device_choice
    m = YOLO(weights_path if weights_path else "yolov8n.pt")
    return m, d

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", use_container_width=True)
    model, d = load_model(weights, device)

    with st.spinner("Running inference..."):
        results = model(image, conf=conf, device=d, verbose=False)
    r = results[0]

    n = 0 if r.boxes is None else r.boxes.shape[0]
    st.metric("Count", n)

    im = r.plot()  # BGR numpy
    st.image(im[..., ::-1], caption="Detections", use_container_width=True)
