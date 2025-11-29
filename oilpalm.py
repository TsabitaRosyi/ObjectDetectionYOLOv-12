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
import matplotlib.pyplot as plt

# =============================
# Fungsi konversi gambar → base64
# =============================
def image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# =============================
# Konfigurasi halaman
# =============================
st.set_page_config(page_title="Deteksi Buah Sawit", layout="wide")

# =============================
# Load Model YOLO
# =============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # ganti sesuai model kamu

model = load_model()

# =============================
# Warna label
# =============================
label_to_color = {
    "matang": Color.RED,
    "mengkal": Color.YELLOW,
    "mentah": Color.BLACK
}
label_annotator = LabelAnnotator()

# =============================
# Fungsi anotasi YOLO 
# =============================
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

            class_name = names[class_id].strip().lower()
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

# =============================
# Crop foto profil
# =============================
def crop_center_square(img):
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    return img.crop((left, top, right, bottom))

# =============================
# Load foto profil
# =============================
profile_img = Image.open("foto.jpg")
profile_img = crop_center_square(profile_img)

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.image("logo.png", width=150)
    st.markdown("<h4>Pilih metode input:</h4>", unsafe_allow_html=True)
    option = st.radio("", ["Upload Gambar", "Upload Video"], label_visibility="collapsed")

    # Created by footer
    st.markdown(
        f"""
        <style>
            .created-by-container {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-top: 20px;
                padding-top: 10px;
                border-top: 1px solid #ccc;
            }}
            .created-by-img {{
                width: 45px;
                height: 45px;
                border-radius: 50%;
                border: 2px solid #444;
                object-fit: cover;
            }}
        </style>

        <div class="created-by-container">
            <img class="created-by-img" src="data:image/png;base64,{image_to_base64(profile_img)}" />
            <div style="font-size:14px; color:#555; font-style:italic;">
                Created by : Tsabit
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================
# Judul Halaman
# =============================
st.markdown("<h1 style='text-align:center;'>🌴 Deteksi Kematangan Buah Sawit</h1>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; font-size:16px; max-width:800px; margin:auto;">
    Sistem menggunakan YOLOv12 untuk mendeteksi tingkat kematangan buah sawit 
    secara otomatis berdasarkan gambar atau video.
</div>
""", unsafe_allow_html=True)

# ==========================================================
# ======================= MODE GAMBAR ======================
# ==========================================================
if option == "Upload Gambar":

    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg","jpeg","png"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        with st.spinner("🔍 Memproses gambar..."):
            results = model(image)
            result_img, class_counts = draw_results(image, results)

        # ======================================================
        # UI RAPI — AREA INPUT & OUTPUT
        # ======================================================
        st.markdown("<br>", unsafe_allow_html=True)
        col_input, col_output = st.columns(2)

        with col_input:
            st.markdown("""
            <div style="border:3px solid black; padding:10px; height:10px;
                        margin-bottom:15px; display:flex; align-items:center;
                        justify-content:center; font-weight:bold; font-size:20px;">
                AREA INPUT FOTO
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col_output:
            st.markdown("""
            <div style="border:3px solid black; padding:10px; height:10px;
                        margin-bottom:15px; display:flex; align-items:center;
                        justify-content:center; font-weight:bold; font-size:20px;">
                AREA HASIL FOTO
            </div>
            """, unsafe_allow_html=True)
            st.image(result_img, use_container_width=True)

        # ==================== DOWNLOAD BUTTON ====================
        buf = BytesIO()
        result_img.save(buf, format="PNG")

        st.download_button(
            "⬇️ Download Hasil Deteksi",
            buf.getvalue(),
            "hasil_deteksi.png",
            "image/png"
        )

        # ==================== REKAP DETEKSI ======================
        total = sum(class_counts.values())

        mentah = class_counts.get("mentah", 0)
        mengkal = class_counts.get("mengkal", 0)
        matang = class_counts.get("matang", 0)

        st.markdown("""
        <div style="margin-top: 20px; border:3px solid black; border-radius:20px; padding:20px;">
        """, unsafe_allow_html=True)

        colA, colB = st.columns([1,2])

        # ==================== TOTAL ====================
        with colA:
            st.markdown("""
            <div style="border:3px solid black; border-radius:20px; padding:10px; 
                        text-align:center; font-weight:bold;">
                Total Deteksi
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f"<h1 style='text-align:center; font-size:60px; margin-top:10px;'>{total}</h1>",
                unsafe_allow_html=True,
            )

        # ==================== DETAIL ====================
        with colB:
            st.markdown("""
            <div style="border:3px solid black; border-radius:20px; padding:15px; 
                        font-size:22px; font-weight:bold;">
            """, unsafe_allow_html=True)

            st.write(f"Mentah  : {mentah}")
            st.write(f"Mengkal : {mengkal}")
            st.write(f"Matang  : {matang}")

            # ==================== Persentase ====================
            if total > 0:
                st.markdown("<br><b>Persentase:</b>", unsafe_allow_html=True)
                st.write(f"Mentah  : {mentah/total*100:.1f}%")
                st.write(f"Mengkal : {mengkal/total*100:.1f}%")
                st.write(f"Matang  : {matang/total*100:.1f}%")

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ==================== PIE CHART ====================
        if total > 0:
            fig, ax = plt.subplots()
            labels = ["Mentah", "Mengkal", "Matang"]
            vals = [mentah, mengkal, matang]

            ax.pie(vals, labels=labels, autopct="%1.1f%%")
            ax.set_title("Distribusi Kematangan")
            st.pyplot(fig)

        # ==================== REKOMENDASI PANEN ====================
        if total > 0:
            st.markdown("<br>", unsafe_allow_html=True)

            if matang > (total * 0.5):
                status = "SIAP PANEN ✅"
                color = "green"
            elif mengkal > (total * 0.5):
                status = "DALAM PROSES — TUNGGU BEBERAPA HARI ⏳"
                color = "orange"
            else:
                status = "BELUM SIAP PANEN ❌"
                color = "red"

            st.markdown(
                f"""
                <div style="border:3px solid {color}; padding:20px;
                            text-align:center; border-radius:15px;
                            font-size:22px; font-weight:bold; color:{color};">
                    {status}
                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================================
# ======================= MODE VIDEO =======================
# ==========================================================
elif option == "Upload Video":

    uploaded_video = st.file_uploader("Unggah Video", type=["mp4","avi","mov"])

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        with st.spinner("🔍 Memproses video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                annotated, _ = draw_results(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), results)
                annotated_bgr = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)

                stframe.image(annotated_bgr, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Video selesai diproses.")
