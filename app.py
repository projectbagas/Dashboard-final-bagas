import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

st.set_page_config(page_title="Skripsi Bagas - Maxim", layout="wide")

# Fungsi Load Assets dengan Pengecekan Path
@st.cache_resource
def load_assets():
    # Pastikan file ada
    for f in ['model_rf.pkl', 'model_xgb.pkl', 'tfidf_vectorizer.pkl', 'maxim_reviews.csv']:
        if not os.path.exists(f):
            st.error(f"File {f} tidak ditemukan di GitHub!")
            st.stop()
            
    df = pd.read_csv('maxim_reviews.csv')
    df['label'] = df['score'].apply(lambda x: 'Puas' if x >= 4 else ('Netral' if x == 3 else 'Tidak Puas'))
    
    rf = joblib.load('model_rf.pkl')
    xgb = joblib.load('model_xgb.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return df, rf, xgb, tfidf

df, rf, xgb, tfidf = load_assets()

# --- SIDEBAR ---
menu = st.sidebar.radio("Menu", ["Dashboard", "Dataset", "Model", "Implementasi"])

if menu == "Dashboard":
    st.title("📊 Dashboard")
    st.bar_chart(df['label'].value_counts())

elif menu == "Dataset":
    st.title("📂 Dataset")
    st.dataframe(df)

elif menu == "Model":
    st.title("🧠 Model")
    st.write("Penjelasan XGBoost dan Random Forest sesuai Bab 3.")

elif menu == "Implementasi":
    st.title("⚖️ Uji Coba")
    # Tabel Akurasi Manual sesuai Skripsi Anda
    st.table(pd.DataFrame({
        'Algoritma': ['XGBoost', 'Random Forest'],
        'Akurasi': ['93%', '80%']
    }))
    
    text = st.text_area("Cek Sentimen:")
    if st.button("Proses"):
        clean = re.sub(r'[^a-z\s]', '', text.lower())
        vec = tfidf.transform([clean])
        res = xgb.predict(vec)[0]
        lbl = {0: "Tidak Puas ❌", 1: "Netral 😐", 2: "Puas ✅"}
        st.success(f"Hasil: {lbl[res]}")