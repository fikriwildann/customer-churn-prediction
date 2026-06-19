import streamlit as st
import joblib
import numpy as np
from datetime import date

st.set_page_config(page_title="Prediksi Customer Churn", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model_churn_rf.joblib")
    scaler_params = joblib.load("scaler_params.joblib")
    top_features = joblib.load("top_features.joblib")
    return model, scaler_params, top_features


model, scaler_params, top_features = load_artifacts()


def scale_value(feature_name, raw_value):
    """Standarisasi nilai mentah memakai mean & std dari StandardScaler hasil training."""
    p = scaler_params[feature_name]
    return (raw_value - p["mean"]) / p["std"]


st.title("📉 Prediksi Customer Churn")
st.write(
    "Aplikasi ini memprediksi kemungkinan seorang pelanggan akan **churn** "
    "(berhenti menggunakan layanan) menggunakan model **Random Forest** yang "
    "telah dioptimasi melalui hyperparameter tuning."
)

st.divider()
st.subheader("Masukkan Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia Pelanggan", min_value=15, max_value=90, value=35)
    total_visits = st.number_input("Total Kunjungan", min_value=0, value=15)
    avg_session_time = st.number_input("Rata-rata Waktu Sesi (menit)", min_value=0.0, value=8.0)
    pages_per_session = st.number_input("Rata-rata Halaman per Sesi", min_value=0.0, value=4.0)
    email_open_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.5)
    email_click_rate = st.slider("Email Click Rate", 0.0, 1.0, 0.25)
    satisfaction_score = st.slider("Skor Kepuasan (1-5)", 1.0, 5.0, 3.5)

with col2:
    total_spent = st.number_input("Total Pengeluaran ($)", min_value=0.0, value=500.0)
    avg_order_value = st.number_input("Rata-rata Nilai Transaksi ($)", min_value=0.0, value=60.0)
    lifetime_value = st.number_input("Lifetime Value ($)", min_value=0.0, value=1200.0)
    marketing_spend_per_user = st.number_input("Biaya Marketing per User ($)", min_value=0.0, value=17.0)
    support_tickets = st.number_input("Jumlah Tiket Dukungan", min_value=0, value=2)

st.divider()
st.subheader("Tanggal Aktivitas")
dcol1, dcol2 = st.columns(2)
with dcol1:
    signup_date = st.date_input("Tanggal Daftar", value=date(2023, 1, 1))
with dcol2:
    last_purchase_date = st.date_input("Tanggal Transaksi Terakhir", value=date.today())

today = date.today()
days_since_signup = (today - signup_date).days
days_since_purchase = (today - last_purchase_date).days
tenure_days = (last_purchase_date - signup_date).days

st.caption(
    f"📅 Dihitung otomatis — Hari sejak daftar: **{days_since_signup}** · "
    f"Hari sejak transaksi terakhir: **{days_since_purchase}** · "
    f"Lama berlangganan: **{tenure_days}** hari"
)

st.divider()

if st.button("🔍 Prediksi Churn", type="primary", use_container_width=True):
    raw_values = {
        "total_spent": total_spent,
        "satisfaction_score": satisfaction_score,
        "support_tickets": support_tickets,
        "avg_session_time": avg_session_time,
        "marketing_spend_per_user": marketing_spend_per_user,
        "pages_per_session": pages_per_session,
        "avg_order_value": avg_order_value,
        "days_since_purchase": days_since_purchase,
        "lifetime_value": lifetime_value,
        "days_since_signup": days_since_signup,
        "tenure_days": tenure_days,
        "email_open_rate": email_open_rate,
        "email_click_rate": email_click_rate,
        "age": age,
        "total_visits": total_visits,
    }

    X_input = np.array([[scale_value(f, raw_values[f]) for f in top_features]])

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.error(f"⚠️ Pelanggan **berpotensi CHURN** (probabilitas: {proba:.1%})")
    else:
        st.success(f"✅ Pelanggan **diprediksi TETAP** (probabilitas churn: {proba:.1%})")

    st.progress(min(max(proba, 0.0), 1.0))

    with st.expander("Lihat detail fitur yang dipakai model"):
        st.json(raw_values)

st.divider()
st.caption(
    "Model: Random Forest (Tuned) · F1-Score ≈ 0.59 pada data uji · "
    "Dibangun untuk UAS Bengkel Koding Data Science — Universitas Dian Nuswantoro"
)
