import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

MODEL_PATH = "model.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.title("😷 Mask / No-Mask Detection (YOLO11)")
st.write("Upload an image. The model will detect faces and predict mask status.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        results = model.predict(
            source=np.array(image),
            conf=0.25,
            save=False
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            st.warning("No faces detected.")
        else:
            annotated_img = result.plot()
            st.image(annotated_img, caption="Prediction Result", use_container_width=True)

            st.subheader("Detections:")
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                label = model.names[cls_id]

                st.write(f"- **{label}** ({conf:.2f}%)")
