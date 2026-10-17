import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Set Judul Web
st.set_page_config(page_title="SPK KNN Diabetes", layout="centered")
st.title("🩺 SPK Prediksi Risiko Diabetes (KNN)")
st.write("Sistem Pendukung Keputusan Berbasis K-Nearest Neighbors")

# Dataset dari Dokumen Tugas
data = {
    'Usia': [25, 30, 35, 40, 45, 50, 55, 28, 38, 48],
    'BMI': [21.5, 23.1, 25.0, 27.5, 30.2, 31.5, 33.0, 22.5, 26.8, 29.5],
    'Glukosa': [90, 95, 105, 120, 140, 155, 165, 92, 115, 145],
    'TekananDarah': [110, 115, 120, 125, 135, 140, 145, 112, 122, 130],
    'RiwayatKeluarga': [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    'Kelas': ['Rendah', 'Rendah', 'Rendah', 'Tinggi', 'Tinggi', 'Tinggi', 'Tinggi', 'Rendah', 'Rendah', 'Tinggi']
}

df = pd.DataFrame(data)

st.subheader("📊 Dataset Referensi")
st.dataframe(df)

# Algoritma KNN Sederhana
X = df[['Usia', 'BMI', 'Glukosa', 'TekananDarah', 'RiwayatKeluarga']]
y = df['Kelas']

k_val = st.slider("Nilai K (Neighbors)", min_value=1, max_value=5, value=3)
knn = KNeighborsClassifier(n_neighbors=k_val)
knn.fit(X, y)

# Input Pasien Baru
st.subheader("📝 Form Input Pasien Baru")
usia = st.number_input("Usia", value=42)
bmi = st.number_input("BMI", value=28.5)
glukosa = st.number_input("Glukosa", value=130)
tekanan = st.number_input("Tekanan Darah", value=128)
riwayat = st.selectbox("Riwayat Keluarga", options=[0, 1], format_func=lambda x: "Ada (1)" if x == 1 else "Tidak Ada (0)")

if st.button("Hitung Prediksi"):
    pasien = np.array([[usia, bmi, glukosa, tekanan, riwayat]])
    prediksi = knn.predict(pasien)[0]
    
    if prediksi == 'Tinggi':
        st.error(f"Hasil Klasifikasi: Risiko **{prediksi}** Diabetes 🔴")
    else:
        st.success(f"Hasil Klasifikasi: Risiko **{prediksi}** Diabetes 🟢")
