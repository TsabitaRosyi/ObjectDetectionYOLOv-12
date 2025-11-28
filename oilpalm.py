import streamlit as st
from PIL import Image
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections
from io import BytesIO
import base64
import tempfile

# Cek apakah streamlit_webrtc tersedia
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    import av
    webrtc_available = True
except ModuleNotFoundError:
    webrtc_available = False


# ===========================
# Fungsi Konversi Base64
# ===========================
def image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# ===========================
# Load Model YOLOv12
# ===========================
st.set_page_config(page_title="Deteksi Buah Sawit - YOLOv12", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("best.pt")  # ← model YOLOv12 kamu

model = load_model()


# ===========================
# Warna Kategori
# ===========================
label_to_color = {
    "Masak": Color.RED,
    "Mengkal": Color.YELLOW,
    "Mentah": Color.BLACK
}

label_annotator = LabelAnnotator()


# ===========================
# FIX YOLOv12 — draw_results()
# ===========================
def draw_results(image, results):
    img = np.array(image.convert("RGB"))
    class_counts = Counter()

    for result in results:
        boxes = result.boxes
        names = result.names  # dict class id → nama label

        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(xyxy, class_ids, confidences):

            # FIX 1: Hindari error jika class_id tidak ada di model
            if class_id not in names:
                print(f"[WARNING] Unknown class id: {class_id}")
                continue

            class_name = names[class_id]

            # FIX 2: Hindari class COCO jika model tidak membawa label custom
            if class_name not in ["Masak", "Mengkal", "Mentah"]:
                print(f"[SKIPPED] Ignoring non-custom class: {class_name}")
                continue

            label = f"{class_name}: {conf:.2f}"
            color = label_to_color.get(class_name, Color.WHITE)
            class_counts[class_name] += 1

            box_annotator = BoxAnnotator(color=color)

            detection = Detections(
                xyxy=np.array([box]),
                confidence=np.array([conf]),
                class_id=np.array([class_id])
            )

            img = box_annotator.annotate(scene=img, detections=detection)
            img = label_annotator.annotate(scene=img, detections=detection, labels=[label])

    return Image.fromarray(img), class_counts


# ===========================
# Sidebar Pilihan
# ===========================
st.title("🥥 Deteksi Kematangan Buah Sawit (YOLOv12)")

option = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Upload Gambar", "Gunakan Kamera (Foto)", "Upload Video", "Kamera Live"]
)


# ===========================
# MODE 1 — Upload Gambar
# ===========================
if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

        with st.spinner("🔍 Memproses gambar..."):
            results = model(image)
            result_img, class_counts = draw_results(image, results)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Gambar Input", use_container_width=True)
        col2.image(result_img, caption="Hasil Deteksi", use_container_width=True)

        st.subheader("Jumlah Objek:")
        for name, count in class_counts.items():
            st.write(f"- **{name}** : {count}")

        buf = BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button("⬇️ Download Hasil", buf.getvalue(), "hasil.png")


# ===========================
# MODE 2 — Kamera Foto
# ===========================
elif option == "Gunakan Kamera (Foto)":
    camera_photo = st.camera_input("Ambil Foto")
    if camera_photo:
        image = Image.open(camera_photo)

        with st.spinner("🔍 Memproses..."):
            results = model(image)
            result_img, class_counts = draw_results(image, results)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Foto Asli")
        col2.image(result_img, caption="Deteksi")


# ===========================
# MODE 3 — Upload Video
# ===========================
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Unggah Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        out = cv2.VideoWriter(
            "output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (int(cap.get(3)), int(cap.get(4)))
        )

        stframe = st.empty()

        with st.spinner("🔍 Memproses video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(rgb)
                annotated_img, _ = draw_results(Image.fromarray(rgb), results)

                out_frame = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
                out.write(out_frame)

                stframe.image(out_frame, channels="BGR")

        cap.release()
        out.release()

        st.download_button("⬇️ Download Video Hasil", open("output.mp4", "rb"), "hasil_video.mp4")


# ===========================
# MODE 4 — Kamera Live
# ===========================
elif option == "Kamera Live":

    if not webrtc_available:
        st.error("⚠️ streamlit-webrtc belum terinstall.")
    else:

        class YOLOv12VideoTransformer(VideoTransformerBase):
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = model(rgb)
                annotated_img, _ = draw_results(Image.fromarray(rgb), results)
                return cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)

        webrtc_streamer(
            key="yolo12-live",
            video_transformer_factory=YOLOv12VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
        )
