import streamlit as st
from PIL import Image
import cv2
import numpy as np
from collections import Counter
from ultralytics import RTDETR
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections
from io import BytesIO
import base64
import tempfile

# Optional import untuk kamera live (biar tidak error di Streamlit Cloud kalau tidak ada streamlit-webrtc)
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    import av
    webrtc_available = True
except ModuleNotFoundError:
    webrtc_available = False

# -----------------------------
# Konversi gambar ke base64
# -----------------------------
def image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# -----------------------------
# Konfigurasi halaman
# -----------------------------
st.set_page_config(page_title="Deteksi Buah Sawit - RT-DETR", layout="wide")

# -----------------------------
# Load model RT-DETR
# -----------------------------
@st.cache_resource
def load_model():
    # Ganti "best.pt" dengan model RT-DETR kamu, misalnya "rtdetr-l.pt"
    return RTDETR("best.pt")

model = load_model()

# -----------------------------
# Warna label
# -----------------------------
label_to_color = {
    "Masak": Color.RED,
    "Mengkal": Color.YELLOW,
    "Mentah": Color.BLACK
}
label_annotator = LabelAnnotator()

# -----------------------------
# Fungsi anotasi deteksi
# -----------------------------
def draw_results(image, results):
    img = np.array(image.convert("RGB"))
    class_counts = Counter()

    for result in results:
        boxes = result.boxes
        names = result.names

        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(xyxy, class_ids, confidences):
            class_name = names[class_id]
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

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image("logo-saraswanti.png", width=150)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h4 style='margin-bottom: 5px;'>Pilih metode input:</h4>", unsafe_allow_html=True)
    options = ["Upload Gambar", "Gunakan Kamera (Foto)", "Upload Video"]
    if webrtc_available:
        options.append("Kamera Live")
    option = st.radio("", options, label_visibility="collapsed")

    # Created by section
    profile_img = Image.open("foto1.jpg")
    st.markdown(
        f"""
        <style>
            .created-by-container {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                margin-top: 15px;
                margin-bottom: 30px;
            }}
            .created-by-img {{
                width: 40px;
                height: 40px;
                border-radius: 50%;
                border: 2px solid #444;
                object-fit: cover;
            }}
            .created-by-text {{
                font-size: 14px;
                color: #555;
                font-style: italic;
                user-select: none;
            }}
        </style>
        <div class="created-by-container">
            <img class="created-by-img" src="data:image/png;base64,{image_to_base64(profile_img)}" alt="Profil" />
            <div class="created-by-text">Created by : hawa tercipta di dunia</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Judul & Deskripsi
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🌴 Deteksi dan Klasifikasi Kematangan Buah Sawit (RT-DETR)</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-size:16px; max-width:800px; margin:auto;">
    Sistem ini menggunakan teknologi RT-DETR untuk mendeteksi dan mengklasifikasikan kematangan buah kelapa sawit 
    secara otomatis berdasarkan gambar atau video input.
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Mode Upload Gambar
# -----------------------------
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

        st.subheader("Jumlah Objek Terdeteksi:")
        for name, count in class_counts.items():
            st.write(f"- **{name}**: {count}")

        buffered = BytesIO()
        result_img.save(buffered, format="PNG")
        st.download_button("⬇️ Download Gambar Hasil Deteksi", buffered.getvalue(), "hasil_deteksi.png", "image/png")

# -----------------------------
# Mode Kamera Foto
# -----------------------------
elif option == "Gunakan Kamera (Foto)":
    camera_photo = st.camera_input("Ambil Foto")
    if camera_photo:
        image = Image.open(camera_photo)
        with st.spinner("🔍 Memproses gambar..."):
            results = model(image)
            result_img, class_counts = draw_results(image, results)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Gambar Input", use_container_width=True)
        col2.image(result_img, caption="Hasil Deteksi", use_container_width=True)

        st.subheader("Jumlah Objek Terdeteksi:")
        for name, count in class_counts.items():
            st.write(f"- **{name}**: {count}")

# -----------------------------
# Mode Upload Video
# -----------------------------
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Unggah Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        stframe = st.empty()
        with st.spinner("🔍 Memproses video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                annotated_frame, _ = draw_results(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), results)
                frame_bgr = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                stframe.image(frame_bgr, channels="BGR", use_container_width=True)

        cap.release()
        out.release()

        with open("output.mp4", "rb") as f:
            st.download_button("⬇️ Download Video Hasil Deteksi", f, file_name="hasil_deteksi.mp4")

# -----------------------------
# Mode Kamera Live (jika tersedia)
# -----------------------------
elif option == "Kamera Live" and webrtc_available:
    class RTDETRVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            annotated_frame, _ = draw_results(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), results)
            return cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

    webrtc_streamer(
        key="rtdetr-live",
        video_transformer_factory=RTDETRVideoTransformer,
        media_stream_constraints={"video": True, "audio": False}
    )
