import pandas as pd
import streamlit as st
from utils.save_to_db import lihat_riwayat

# Ambil 20 data terakhir dari database
data = lihat_riwayat(20)

# Buat DataFrame dengan nama kolom sesuai tabel
df = pd.DataFrame(data, columns=[
    "id", "sumber", "review", "label", "confidence", "prob_cg", "prob_or", "waktu"
])

st.title("Riwayat Deteksi Ulasan")
st.dataframe(df)