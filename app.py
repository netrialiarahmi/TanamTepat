import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import requests
import os
from pathlib import Path

# URL file model
MODEL_URL = "https://github.com/netrialiarahmi/TanamTepat/releases/download/agriculture/Model_Fix.h5"

def download_model(url, save_path="Model_Fix.h5"):
    """Mengunduh file model dari URL dan menyimpannya secara lokal."""
    try:
        if not Path(save_path).exists():
            with st.spinner("Mengunduh model..."):
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers, stream=True)
                response.raise_for_status()  # Raise error jika status code tidak 200
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            st.success("Model berhasil diunduh.")
        else:
            st.info("Model sudah tersedia secara lokal.")
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal mengunduh model: {e}")
        st.stop()
    return save_path

def load_model():
    """Memuat model dari file lokal."""
    model_path = download_model(MODEL_URL)
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

def predict_soil_type(model, image):
    """Memproses gambar dan memprediksi jenis tanah."""
    try:
        image = image.resize((224, 224))  # Asumsi model menerima input ukuran 224x224
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
        predictions = model.predict(image_array)
        soil_types = ['Tanah Aluvial', 'Tanah Hitam', 'Tanah Liat', 'Tanah Merah']
        return soil_types[np.argmax(predictions)]
    except Exception as e:
        st.error(f"Kesalahan saat memprediksi: {e}")
        return None

def get_plant_recommendations(soil_type):
    """Memberikan rekomendasi tanaman berdasarkan jenis tanah."""
    recommendations = {
        'Tanah Aluvial': [
            {'name': 'Padi', 'image': 'https://example.com/padi.jpg'},
            {'name': 'Gandum', 'image': 'https://example.com/gandum.jpg'},
            {'name': 'Tebu', 'image': 'https://example.com/tebu.jpg'},
            {'name': 'Kapas', 'image': 'https://example.com/kapas.jpg'}
        ],
        'Tanah Hitam': [
            {'name': 'Kapas', 'image': 'https://example.com/kapas.jpg'},
            {'name': 'Kacang Tanah', 'image': 'https://example.com/kacang.jpg'},
            {'name': 'Bunga Matahari', 'image': 'https://example.com/sunflower.jpg'},
            {'name': 'Millet', 'image': 'https://example.com/millet.jpg'}
        ],
        'Tanah Liat': [
            {'name': 'Padi', 'image': 'https://example.com/padi.jpg'},
            {'name': 'Jute', 'image': 'https://example.com/jute.jpg'},
            {'name': 'Tebu', 'image': 'https://example.com/tebu.jpg'}
        ],
        'Tanah Merah': [
            {'name': 'Millet', 'image': 'https://example.com/millet.jpg'},
            {'name': 'Kacang-kacangan', 'image': 'https://example.com/beans.jpg'},
            {'name': 'Kacang Tanah', 'image': 'https://example.com/kacang.jpg'},
            {'name': 'Kentang', 'image': 'https://example.com/potato.jpg'}
        ]
    }
    return recommendations.get(soil_type, [])

def main():
    st.set_page_config(page_title="Deteksi Jenis Tanah dan Rekomendasi Tanaman", layout="centered")
    st.markdown("<style>.css-18e3th9 {padding-top: 2rem;} .css-1d391kg {padding: 1rem;}</style>", unsafe_allow_html=True)
    st.title("🌱 Deteksi Jenis Tanah dan Rekomendasi Tanaman")

    # Memuat model
    model = load_model()

    # Unggah gambar
    uploaded_image = st.file_uploader("Unggah gambar tanah Anda", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_image)
        st.image(image, caption="📸 Gambar Tanah yang Diunggah", use_column_width=True)

        # Prediksi jenis tanah
        soil_type = predict_soil_type(model, image)
        if soil_type:
            st.markdown(f"### 🌍 Jenis Tanah: **{soil_type}**")

            # Dapatkan rekomendasi tanaman
            plants = get_plant_recommendations(soil_type)
            st.markdown("### 🌿 Rekomendasi Tanaman:")

            col1, col2 = st.columns(2)
            for i, plant in enumerate(plants):
                with col1 if i % 2 == 0 else col2:
                    st.image(plant['image'], caption=plant['name'], use_column_width=True)

            # Tambahkan deskripsi tambahan
            st.markdown("---")
            st.markdown(f"Tanah **{soil_type}** dikenal dengan ciri khas tertentu yang cocok untuk tanaman-tanaman tersebut.")

    # Footer
    st.markdown("---")
    st.markdown("Dibuat dengan ❤️ untuk membantu petani.")

if __name__ == "__main__":
    main()
