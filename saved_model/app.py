import streamlit as st
import pandas as pd
import re
import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# -------------------------------
# CSS: Atur posisi & ukuran navigasi tab
# -------------------------------
st.markdown("""
<style>
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
required_files = ['model.safetensors', 'config.json', 'vocab.txt', 'tokenizer_config.json', 'special_tokens_map.json']
missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
if missing_files:
    st.error(f"File berikut tidak ditemukan di {model_path}: {', '.join(missing_files)}")
    st.stop()

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
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
    
    st.title("Selamat Datang di Website Deteksi Ulasan Palsu")
    st.write("---------------------------------------------🔍 Cerdas Memilah, Tepat Memilih 🔍-----------------------------------------")
    st.write("""
             
    Website ini akan membantu anda mendeteksi apakah sebuah ulasan merupakan ulasan asli atau palsu dengan mengklasifikasikan ulasan pada kategori:
    
    - **OR (Asli)**
    - **CG (Palsu)**
    
    Fitur utama:
    - Deteksi ulasan secara manual/langsung atau mengunggah file CSV berisi kumpulan ulasan
    - Menampilkan visualisasi probabilitas
    - Mengunduh hasil klasifikasi
    
    Silakan klik tab **Deteksi** untuk mulai.
    """)

# -------------------------------
# Halaman: Deteksi
# -------------------------------
with tabs[1]:
    st.title("Deteksi Ulasan Palsu dengan DistilBERT")

    input_option = st.radio("Pilih metode input:", ("Input Teks Manual", "Unggah File CSV"))

    if input_option == "Input Teks Manual":
        user_input = st.text_area("Masukkan Ulasan:", height=150)
        if st.button("Prediksi"):
            if user_input.strip() == "":
                st.warning("Harap masukkan ulasan teks!")
            else:
                label, confidence, probs = predict_review(user_input, model, tokenizer, device)
                st.subheader("Hasil Prediksi:")
                st.write(f"**Label:** {label}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write(f"**Probabilitas OR (Asli):** {probs[1]*100:.2f}%")
                st.write(f"**Probabilitas CG (Palsu):** {probs[0]*100:.2f}%")

                st.subheader("Visualisasi Probabilitas")
                probs_df = pd.DataFrame({
                    'Kelas': ['CG (Palsu)', 'OR (Asli)'],
                    'Probabilitas': [probs[0]*100, probs[1]*100]
                })
                st.bar_chart(probs_df.set_index('Kelas'))

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

                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Unduh Hasil Prediksi sebagai CSV",
                        data=csv,
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error saat memproses file CSV: {str(e)}")

# -------------------------------
# Halaman: Tentang
# -------------------------------
with tabs[2]:
    st.title("Tentang Aplikasi")
    st.markdown("""
    Ulasan produk memainkan peran yang sangat penting dalam memengaruhi keputusan pembelian, terutama dalam platform e-commerce. Ulasan yang ditulis oleh pelanggan sebelumnya memberikan gambaran nyata mengenai kualitas, performa, dan kepuasan terhadap suatu produk. Dalam konteks ini, ulasan berfungsi sebagai referensi yang membantu calon pembeli untuk menilai apakah produk tersebut layak dibeli. Oleh karena itu, keaslian dan kredibilitas ulasan menjadi faktor krusial dalam menjaga kepercayaan konsumen.

Sayangnya, kemunculan ulasan palsu menjadi tantangan besar dalam dunia digital. Ulasan palsu sering kali dibuat untuk tujuan manipulatif, baik untuk meningkatkan penjualan produk tertentu secara tidak jujur, maupun untuk menjatuhkan reputasi kompetitor. Hal ini dapat merugikan pembeli dan merusak integritas ekosistem e-commerce secara keseluruhan.

Untuk menjawab permasalahan tersebut, dikembangkanlah sebuah sistem deteksi ulasan palsu berbasis machine learning menggunakan model DistilBERT. Sistem ini dirancang untuk secara otomatis mengklasifikasikan teks ulasan ke dalam dua kategori, yaitu OR (Original) dan CG (Computer Generated). Aplikasi ini dibangun menggunakan teknologi Python dan Streamlit, serta didukung oleh pustaka NLP dari Hugging Face. Dengan adanya sistem ini, diharapkan proses identifikasi ulasan yang tidak autentik dapat dilakukan secara lebih efisien, sehingga pengguna dapat mengambil keputusan berdasarkan informasi yang lebih valid dan terpercaya.

                
""")
