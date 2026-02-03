import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# 1. SET PAGE CONFIG
st.set_page_config(page_title="Skripsi Bagas - Maxim Analisis", layout="wide")

# 2. FUNGSI LOAD ASSETS (Sesuai cara aman yang kamu mau)
@st.cache_resource
def load_assets():
    # List file yang wajib ada di GitHub
    required_files = ['model_rf.pkl', 'model_xgb.pkl', 'tfidf_vectorizer.pkl', 'maxim_reviews.csv']
    
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"⚠️ File **{f}** tidak ditemukan di repositori GitHub!")
            st.stop()
            
    # Load Data
    df = pd.read_csv('maxim_reviews.csv')
    df['label'] = df['score'].apply(lambda x: 'Puas' if x >= 4 else ('Netral' if x == 3 else 'Tidak Puas'))
    
    # Load Models menggunakan joblib (Saran terbaik untuk Cloud)
    rf_model = joblib.load('model_rf.pkl')
    xgb_model = joblib.load('model_xgb.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    
    return df, rf_model, xgb_model, tfidf

# Menjalankan fungsi load
df, rf_model, xgb_model, tfidf = load_assets()

# --- HEADER ---
st.title("📊 Dashboard Analisis Kepuasan Pengguna Maxim")
st.markdown("Oleh: **Bagas Dwi Ardianto** (217006516109)")
st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Menu Navigasi")
    menu = st.radio("Pilih Halaman:", ["📈 Statistik & Dataset", "⚖️ Perbandingan Model", "🔍 Live Sentiment Test"])
    st.divider()
    st.info("Gunakan menu ini untuk menavigasi bagian-bagian skripsi.")

# --- HALAMAN 1: STATISTIK & DATASET ---
if menu == "📈 Statistik & Dataset":
    tab1, tab2 = st.tabs(["📊 Distribusi Sentimen", "📂 View Raw Data"])
    
    with tab1:
        st.subheader("Distribusi Sentimen Pengguna")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            # Menggunakan visualisasi Seaborn agar lebih cantik
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(x='label', data=df, palette='viridis', ax=ax)
            st.pyplot(fig)
        with col_b:
            st.write("### Ringkasan Data")
            st.write(f"**Total Data:** {len(df)} ulasan")
            st.write("**Sumber:** Google Play Store")
            st.write("**Label:** Puas, Netral, Tidak Puas")
            
    with tab2:
        st.subheader("Tampilan Dataset Mentah")
        st.dataframe(df[['userName', 'score', 'content', 'label']], use_container_width=True)

# --- HALAMAN 2: PERBANDINGAN MODEL ---
elif menu == "⚖️ Perbandingan Model":
    st.subheader("Hasil Evaluasi Algoritma (Sesuai Bab 4)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Akurasi XGBoost", "93%", delta="Unggul")
    with c2:
        st.metric("Akurasi Random Forest", "80%", delta="-13%", delta_color="inverse")
    
    st.divider()
    
    st.write("### Tabel Metrik Klasifikasi")
    metrics_df = pd.DataFrame({
        'Metrik': ['Presisi', 'Recall', 'F1-Score'],
        'XGBoost': [0.88, 0.93, 0.90],
        'Random Forest': [0.73, 0.80, 0.77]
    })
    st.table(metrics_df)
    
    with st.expander("Lihat Penjelasan Algoritma"):
        st.write("""
        - **XGBoost:** Algoritma boosting yang fokus pada perbaikan error di tiap iterasi.
        - **Random Forest:** Algoritma bagging yang membangun banyak decision tree secara paralel.
        """)

# --- HALAMAN 3: LIVE SENTIMENT TEST ---
elif menu == "🔍 Live Sentiment Test":
    st.subheader("🔍 Uji Coba Prediksi Sentimen")
    st.write("Masukkan teks ulasan di bawah ini untuk melihat hasil prediksi algoritma.")
    
    raw_text = st.text_area("Masukkan ulasan pelanggan di sini:", placeholder="Contoh: Aplikasinya sangat membantu dan cepat...")
    
    col1, col2 = st.columns(2)
    with col1:
        method = st.radio("Pilih Algoritma Prediksi:", ("XGBoost", "Random Forest"))
    
    if st.button("🚀 Analisis Sekarang"):
        if raw_text:
            # 1. Preprocessing
            clean_input = re.sub(r'[^a-z\s]', '', raw_text.lower())
            
            # 2. Transform ke TF-IDF
            vec_input = tfidf.transform([clean_input])
            
            # 3. Predict
            if method == "XGBoost":
                res = xgb_model.predict(vec_input)[0]
            else:
                res = rf_model.predict(vec_input)[0]
            
            # 4. Map Result (Mapping: 0=Tidak Puas, 1=Netral, 2=Puas)
            # Pastikan urutan angka ini sesuai dengan saat kamu training di Colab
            labels = {0: "Tidak Puas ❌", 1: "Netral 😐", 2: "Puas ✅"}
            
            st.divider()
            st.markdown(f"### Hasil Prediksi ({method}):")
            st.success(f"**{labels[res]}**")
        else:
            st.warning("⚠️ Silahkan masukkan teks ulasan terlebih dahulu.")
