from utils.save_to_db import init_db, save_detection_record

import streamlit as st
import pandas as pd
import re
import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Buat database dan tabel
init_db()

# -------------------------------
# CSS: Komposisi warna dan elemen visual
# -------------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #fff !important;
}
body, .stApp, .css-10trblm, .css-1v0mbdj, .css-1d391kg, .css-ffhzg2, .css-1cpxqw2,
.stMarkdown, .stText, .stHeader, .stSubheader, .stDataFrame, .stRadio, .stDownloadButton, .stFileUploader,
.stTextInput, .stTextArea, label, .st-bw, .st-c3, .st-c4, .st-c5, .st-c6, .st-c7, .st-c8, .st-c9, .st-ca, .st-cb, .st-cc, .st-cd, .st-ce, .st-cf, .st-cg, .st-ch, .st-ci, .st-cj, .st-ck, .st-cl, .st-cm, .st-cn, .st-co, .st-cp, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
    color: #111 !important;
}
/* Button utama dengan aksen biru */
.stButton>button, .stDownloadButton>button {
    color: #fff !important;
    background-color: #2E86C1 !important;
    border: 1px solid #2E86C1 !important;
    font-weight: bold;
    border-radius: 6px;
    transition: 0.2s;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background-color: #117A65 !important;
    border: 1px solid #117A65 !important;
}
/* Input dan textarea */
.stTextInput input, .stTextArea textarea {
    color: #111 !important;
    background-color: #fff !important;
}
/* Navigasi tab di kanan */
[data-baseweb="tab-list"] {
    justify-content: flex-end !important;
    margin-top: 10px;
    margin-right: 20px;
}
/* Jarak antar tab */
[data-baseweb="tab"] {
    margin-right: 24px !important;
}
/* Ukuran besar dan tebal untuk font tab */
div[role="tab"] {
    font-size: 32px !important;
    font-weight: 700 !important;
    padding: 12px 20px !important;
}
/* Card panel */
.panel {
    background-color: #F4F8FB;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Fungsi Pembersihan dan Prediksi
# -------------------------------
def cleaning(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = text.strip()
    return text

def predict_review(text, model, tokenizer, device):
    cleaned_text = cleaning(text)
    if not cleaned_text:
        return 'Tidak Valid', 0.0, [0.0, 0.0]

    encodings = tokenizer(cleaned_text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    encodings = {key: val.to(device) for key, val in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = logits.argmax(-1).cpu().numpy()[0]

    label = 'OR (Asli)' if prediction == 1 else 'CG (Palsu)'
    confidence = probs[prediction] * 100
    return label, confidence, probs

def predict_from_csv(df, model, tokenizer, device):
    results = []
    for text in df['review']:
        label, confidence, probs = predict_review(text, model, tokenizer, device)
        results.append({
            'Review': text,
            'Label': label,
            'Confidence (%)': confidence,
            'Probabilitas CG (%)': probs[0] * 100,
            'Probabilitas OR (%)': probs[1] * 100
        })
    return pd.DataFrame(results)

# -------------------------------
# Load Model & Tokenizer
# -------------------------------
model_path = 'saved_model'
required_files = [
    'model.safetensors', 'config.json',
    'vocab.txt', 'tokenizer_config.json',
    'special_tokens_map.json'
]

missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
if missing_files:
    st.error(f"File berikut tidak ditemukan di {model_path}: {', '.join(missing_files)}")
    st.stop()

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    ).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model.eval()
except Exception as e:
    st.error(f"Gagal memuat model atau tokenizer: {str(e)}")
    st.stop()

# -------------------------------
# Navigasi Tab Horizontal (di kanan)
# -------------------------------
tabs = st.tabs(["Beranda", "Deteksi", "Tentang"])

# -------------------------------
# Halaman: Beranda
# -------------------------------
with tabs[0]:
    st.markdown("""
    <div style='text-align:center; margin-bottom: 10px;'>
        <img src="https://png.pngtree.com/png-vector/20230122/ourmid/pngtree-cartoon-online-shopping-icon-on-white-background-splash-shopping-checkout-vector-png-image_49384152.jpg" width="100" style="display:block; margin-left:auto; margin-right:auto;">
        <h1 style='color:#2E86C1; margin-bottom:0;'>VeriView</h1>
        <span style='color:#117A65; font-size:20px; display:block;'>Deteksi Ulasan Palsu Otomatis</span>
    </div>
    <hr style='border:2px solid #2E86C1;'>
    """, unsafe_allow_html=True)
        
    st.markdown("""
    <div style='background:#F4F8FB; border-radius:10px; padding:18px; margin-bottom:18px; border:1px solid #e0e0e0'>
        <h3 style='text-align:center; color:#2E86C1; margin-bottom:10px;'>🔍 Cerdas Memilah, Tepat Memilih 🔍</h3>
        <p style='text-align:justify; color:#111;'>
            Website ini akan membantu anda mendeteksi apakah sebuah ulasan merupakan ulasan <b>asli</b> atau <b>palsu</b> dengan mengklasifikasikan ulasan pada kategori:<br>
            <b>- OR (Asli)</b><br>
            <b>- CG (Palsu)</b>
        </p>
        <b>Fitur utama:</b>
        <ul>
            <li>Deteksi ulasan secara manual/langsung atau mengunggah file CSV berisi kumpulan ulasan</li>
            <li>Menampilkan visualisasi probabilitas</li>
            <li>Mengunduh hasil klasifikasi</li>
        </ul>
                <b>Silakan klik tab <span style='color:#2E86C1;'>Deteksi</span> untuk mulai.</b>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Halaman: Deteksi
# -------------------------------
with tabs[1]:
    st.markdown("<h2 style='color:#2E86C1;'>Deteksi Ulasan dengan VeriView</h2>", unsafe_allow_html=True)
    input_option = st.radio("Pilih metode input:", ("Input Teks Manual", "Unggah File CSV"))

    if input_option == "Input Teks Manual":
        user_input = st.text_area("Masukkan teks ulasan :", height=150, placeholder="Tulis ulasan di sini...")
        if st.button("Prediksi"):
            if user_input.strip() == "":
                st.warning("Harap masukkan teks ulasan!")
            else:
                label, confidence, probs = predict_review(user_input, model, tokenizer, device)
                st.subheader("Hasil Prediksi:")
                st.write(f"**Label:** <span style='color:#2E86C1'>{label}</span>", unsafe_allow_html=True)
                st.write(f"**Confidence:** <span style='color:#117A65'>{confidence:.2f}%</span>", unsafe_allow_html=True)
                st.write(f"**Probabilitas OR (Asli):** <span style='color:#117A65'>{probs[1]*100:.2f}%</span>", unsafe_allow_html=True)
                st.write(f"**Probabilitas CG (Palsu):** <span style='color:#884EA0'>{probs[0]*100:.2f}%</span>", unsafe_allow_html=True)

                st.subheader("Visualisasi Probabilitas")
                probs_df = pd.DataFrame({
                    'Kelas': ['CG (Palsu)', 'OR (Asli)'],
                    'Probabilitas': [probs[0]*100, probs[1]*100]
                })
                st.bar_chart(probs_df.set_index('Kelas'))

                # ✅ Simpan ke DB
                save_detection_record(user_input, label, confidence, probs[0]*100, probs[1]*100, sumber="manual")
                st.success("Hasil prediksi telah disimpan ke database.")
                
    else:
        st.subheader("Unggah File CSV")
        uploaded_file = st.file_uploader("Pilih file CSV (harus memiliki kolom 'review')", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'review' not in df.columns:
                    st.error("File CSV harus memiliki kolom 'review'!")
                else:
                    result_df = predict_from_csv(df, model, tokenizer, device)
                    st.subheader("Hasil Prediksi")
                    st.dataframe(result_df)

                    st.subheader("Distribusi Kelas Prediksi")
                    class_counts = result_df['Label'].value_counts()
                    class_counts_df = pd.DataFrame({
                        'Kelas': class_counts.index,
                        'Jumlah': class_counts.values
                    })
                    st.bar_chart(class_counts_df.set_index('Kelas'))

                    # ✅ Simpan semua ke DB
                    for row in result_df.itertuples():
                        save_detection_record(row.Review, row.Label, row._3, row._4, row._5, sumber="csv")

                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Unduh Hasil Prediksi sebagai CSV",
                        data=csv,
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error saat memproses file CSV: {str(e)}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Halaman: Tentang
# -------------------------------
with tabs[2]:
    st.markdown("<h2 style='color:#2E86C1; '>Tentang VeriView</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align:justify; color:#111;'>
    Ulasan produk memainkan peran yang sangat penting dalam memengaruhi keputusan pembelian, terutama dalam platform e-commerce. Ulasan yang ditulis oleh pelanggan sebelumnya memberikan gambaran nyata mengenai kualitas, performa, dan kepuasan terhadap suatu produk. Dalam konteks ini, ulasan berfungsi sebagai referensi yang membantu calon pembeli untuk menilai apakah produk tersebut layak dibeli. Oleh karena itu, keaslian dan kredibilitas ulasan menjadi faktor krusial dalam menjaga kepercayaan konsumen.<br><br>
    Sayangnya, kemunculan ulasan palsu menjadi tantangan besar dalam dunia digital. Ulasan palsu sering kali dibuat untuk tujuan manipulatif, baik untuk meningkatkan penjualan produk tertentu secara tidak jujur, maupun untuk menjatuhkan reputasi kompetitor. Hal ini dapat merugikan pembeli dan merusak integritas ekosistem e-commerce secara keseluruhan.<br><br>
    Untuk menjawab permasalahan tersebut, dikembangkanlah sebuah sistem deteksi ulasan palsu berbasis machine learning menggunakan model DistilBERT. Sistem ini dirancang untuk secara otomatis mengklasifikasikan teks ulasan ke dalam dua kategori, yaitu OR (Original) dan CG (Computer Generated). Website ini dibangun menggunakan teknologi Python dan Streamlit, serta didukung oleh pustaka NLP dari Hugging Face. Dengan adanya sistem ini, diharapkan proses identifikasi ulasan yang tidak autentik dapat dilakukan secara lebih efisien, sehingga pengguna dapat mengambil keputusan berdasarkan informasi yang lebih valid dan terpercaya.
    </div>
    <i>Semoga membantu Anda mengambil keputusan yang lebih cerdas dan terpercaya!</i>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<hr>
<div style='text-align:center; color:gray; font-size:14px'>
    &copy; 2025 <b style='color:#2E86C1'>VeriView</b> | Dibuat dengan ❤️ oleh nistiaen
</div>
""", unsafe_allow_html=True)
