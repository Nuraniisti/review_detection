import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan tokenizer (dilatih dengan epoch terbaik berdasarkan evaluasi)
@st.cache_resource

def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("./saved_model")
    tokenizer = DistilBertTokenizer.from_pretrained("./saved_model")
    return model, tokenizer

model, tokenizer = load_model()

# Function cleaning (tanpa lowercase)
def clean_text(text):
    import re
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.strip()
    return text

# Fungsi prediksi
@st.cache_data

def predict(texts):
    cleaned = [clean_text(t) for t in texts]
    encoding = tokenizer(cleaned, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs, cleaned

# Sidebar
st.sidebar.title("Menu")
menu = st.sidebar.radio("Navigasi", ["Home", "Deteksi Ulasan"])

if menu == "Home":
    st.title("Deteksi Ulasan Palsu dengan DistilBERT")
    st.markdown("""
    Aplikasi ini memungkinkan pengguna untuk:
    - Memasukkan ulasan secara manual atau mengunggah file CSV
    - Melakukan pre-processing (cleaning dan tokenizing)
    - Melihat hasil prediksi: label (OR/CG), probabilitas, dan confidence
    - Melihat visualisasi hasil
    - Mengunduh hasil prediksi

    Catatan: Model yang digunakan telah dilatih dan diuji pada beberapa skenario jumlah epoch (20, 50, dan 100) dan memilih model terbaik berdasarkan metrik evaluasi (f1-score dan akurasi). Versi model yang digunakan di aplikasi ini adalah model terbaik dari skenario tersebut.
    """)

elif menu == "Deteksi Ulasan":
    st.title("Deteksi Ulasan")

    input_mode = st.radio("Pilih metode input", ["Manual", "Upload File"])

    if input_mode == "Manual":
        user_input = st.text_area("Masukkan ulasan:")
        if st.button("Deteksi") and user_input:
            preds, probs, cleaned = predict([user_input])
            label = "OR (Asli)" if preds[0] == 1 else "CG (Palsu)"
            st.write("### Hasil Deteksi")
            st.write(f"Teks Cleaning: {cleaned[0]}")
            st.write(f"Label: **{label}**")
            st.write(f"Confidence Score: {np.max(probs[0]):.4f}")
            st.write(f"Probabilitas OR: {probs[0][1]:.4f}")
            st.write(f"Probabilitas CG: {probs[0][0]:.4f}")

    elif input_mode == "Upload File":
        file = st.file_uploader("Unggah file CSV", type=["csv"])
        if file is not None:
            df = pd.read_csv(file)
            if 'review' in df.columns:
                if st.button("Deteksi Batch"):
                    preds, probs, cleaned = predict(df['review'].tolist())
                    result_df = pd.DataFrame({
                        'review': df['review'],
                        'cleaned': cleaned,
                        'label': ["OR" if p == 1 else "CG" for p in preds],
                        'confidence': np.max(probs, axis=1),
                        'prob_OR': probs[:, 1],
                        'prob_CG': probs[:, 0]
                    })
                    st.write("### Hasil Deteksi Batch")
                    st.dataframe(result_df)

                    st.write("### Visualisasi Distribusi Prediksi")
                    fig, ax = plt.subplots()
                    sns.countplot(x='label', data=result_df, palette='pastel', ax=ax)
                    ax.set_title('Distribusi Hasil Prediksi')
                    st.pyplot(fig)

                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Unduh Hasil sebagai CSV", data=csv, file_name="hasil_deteksi.csv", mime="text/csv")
            else:
                st.warning("Kolom 'review' tidak ditemukan dalam file.")