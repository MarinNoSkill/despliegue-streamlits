# -*- coding: utf-8 -*-
"""
Despliegue - Predicción de Ataque al Corazón
Modelo: Random Forest (modelo-cla.pkl)
Ejecutar:  streamlit run Despliegue_ValidacionCruz_Clasif.py
"""
 
# ------------------------------------------------------------
# Librerías
# ------------------------------------------------------------
import pickle
import numpy as np
import pandas as pd
import streamlit as st
 
# ------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Ataque al Corazón",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)
 
# ------------------------------------------------------------
# Cargar el modelo
# ------------------------------------------------------------
filename = "modelo-cla.pkl"
modelo, labelencoder, variables, min_max_scaler = pickle.load(open(filename, "rb"))
 
# ------------------------------------------------------------
# ESTILOS  (Google Fonts + fondo animado + glassmorphism)
# ------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
 
    <style>
    /* ---------- Fondo con gradiente animado ---------- */
    .stApp {
        background: linear-gradient(-45deg, #0b0f1a, #1a1035, #3a0d3f, #16213e);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
        font-family: 'Inter', sans-serif;
        color: #eef1f7;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
 
    /* Oculta el header por defecto de Streamlit para look más limpio */
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
 
    /* ---------- Tarjeta de título (hero) ---------- */
    .hero {
        position: relative;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 24px;
        padding: 38px 34px 30px 34px;
        text-align: center;
        margin-bottom: 30px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.35);
        overflow: hidden;
        animation: fadeInUp 0.9s ease both;
    }
    /* brillo que cruza la tarjeta */
    .hero::before {
        content: "";
        position: absolute; top: -50%; left: -60%;
        width: 60%; height: 200%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.14), transparent);
        transform: rotate(18deg);
        animation: shine 6s ease-in-out infinite;
    }
    @keyframes shine {
        0%   { left: -60%; }
        60%  { left: 130%; }
        100% { left: 130%; }
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.3rem;
        letter-spacing: -0.5px;
        margin: 0 0 8px 0;
        background: linear-gradient(90deg, #ff6a88, #ff99ac, #a18cd1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        color: #b8c0d8;
        font-size: 1.02rem;
        margin: 0;
        font-weight: 400;
    }
 
    /* corazón latiendo hecho con CSS (sin emoji) */
    .heart {
        width: 30px; height: 30px;
        background: #ff4d6d;
        transform: rotate(-45deg);
        margin: 0 auto 14px auto;
        position: relative;
        animation: beat 1.3s ease-in-out infinite;
        box-shadow: 0 0 24px rgba(255,77,109,0.6);
    }
    .heart::before, .heart::after {
        content: ""; width: 30px; height: 30px;
        background: #ff4d6d; border-radius: 50%; position: absolute;
    }
    .heart::before { top: -15px; left: 0; }
    .heart::after  { top: 0; left: 15px; }
    @keyframes beat {
        0%, 100% { transform: rotate(-45deg) scale(1); }
        25%      { transform: rotate(-45deg) scale(1.18); }
        40%      { transform: rotate(-45deg) scale(1); }
    }
 
    /* ---------- Sección de formulario ---------- */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.15rem;
        color: #ffffff;
        margin: 6px 0 4px 2px;
        letter-spacing: 0.3px;
    }
 
    /* Labels de inputs */
    label, .stSelectbox label, .stSlider label {
        color: #dfe4f0 !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }
 
    /* Cajas de selectbox */
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #fff !important;
    }
 
    /* Color del slider */
    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background: #ff6a88 !important;
        border: 3px solid #fff !important;
        box-shadow: 0 0 12px rgba(255,106,136,0.7) !important;
    }
 
    /* ---------- Botón ---------- */
    div.stButton > button {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 1.4rem;
        width: 100%;
        margin-top: 10px;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(255,65,108,0.4);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 12px 30px rgba(255,65,108,0.6);
    }
    div.stButton > button:active { transform: translateY(-1px) scale(0.99); }
 
    /* ---------- Cajas de resultado ---------- */
    .result {
        border-radius: 18px;
        padding: 26px 28px;
        text-align: center;
        margin-top: 24px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        animation: fadeInUp 0.6s ease both;
        backdrop-filter: blur(10px);
    }
    .result .label { font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase; opacity: 0.8; font-family:'Inter',sans-serif; font-weight:600; }
    .result .big   { font-size: 1.6rem; margin-top: 6px; display:block; }
 
    .result-danger {
        background: rgba(255,60,80,0.14);
        border: 1px solid rgba(255,60,80,0.5);
        color: #ff9fb0;
        box-shadow: 0 0 30px rgba(255,60,80,0.25);
    }
    .result-safe {
        background: rgba(46,213,115,0.13);
        border: 1px solid rgba(46,213,115,0.5);
        color: #7bffb0;
        box-shadow: 0 0 30px rgba(46,213,115,0.2);
    }
 
    /* aviso de error del modelo */
    .note {
        background: rgba(255,193,7,0.10);
        border-left: 3px solid #ffc107;
        border-radius: 10px;
        padding: 12px 16px;
        color: #ffe08a;
        font-size: 0.9rem;
        margin-bottom: 22px;
        font-family: 'Inter', sans-serif;
    }
 
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="heart"></div>
        <h1>Predicción de Ataque al Corazón</h1>
        <p>Modelo Random Forest &nbsp;·&nbsp; ingresa los datos del paciente</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# Aviso del error del modelo
st.markdown(
    '<div class="note"><b>Nota:</b> el modelo tiene un error aproximado del 8% '
    "(MAPE). El resultado es una estimación, no un diagnóstico médico.</div>",
    unsafe_allow_html=True,
)
 
# ------------------------------------------------------------
# Interfaz de captura de datos
# ------------------------------------------------------------
st.markdown('<div class="section-title">Datos del paciente</div>', unsafe_allow_html=True)
 
col1, col2 = st.columns(2)
 
with col1:
    age = st.slider("Edad", min_value=1, max_value=82, value=45, step=1)
    hypertension = st.selectbox("Hipertensión", ["No", "Yes"])
    heart_disease = st.selectbox("Enfermedad cardíaca", ["No", "Yes"])
 
with col2:
    avg_glucose_level = st.slider(
        "Nivel promedio de glucosa", min_value=55.0, max_value=272.0, value=100.0, step=0.1
    )
    ever_married = st.selectbox("Alguna vez casado/a", ["No", "Yes"])
    smoking_status = st.selectbox(
        "Estado de fumador",
        ["'never smoked'", "'formerly smoked'", "smokes", "Unknown"],
    )
 
# Dataframe con los mismos nombres de variables del entrenamiento
datos = [[age, avg_glucose_level, hypertension, heart_disease, ever_married, smoking_status]]
data = pd.DataFrame(
    datos,
    columns=["age", "avg_glucose_level", "hypertension", "heart_disease", "ever_married", "smoking_status"],
)
 
# ------------------------------------------------------------
# Preparación de datos (igual que en el entrenamiento)
# ------------------------------------------------------------
data_preparada = data.copy()
 
# En despliegue drop_first = False
data_preparada = pd.get_dummies(
    data_preparada,
    columns=["smoking_status", "hypertension", "heart_disease", "ever_married"],
    drop_first=False,
    dtype=int,
)
 
# Se añaden las columnas faltantes y se ordenan como en el entrenamiento
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
 
# Normalización: como el modelo final es Random Forest (árboles) NO se normaliza.
# Si usaras KNN, Red, SVM o Regresión, descomenta la línea siguiente:
# data_preparada[['age','avg_glucose_level']] = min_max_scaler.transform(data_preparada[['age','avg_glucose_level']])
 
# ------------------------------------------------------------
# Predicción (al presionar el botón)
# ------------------------------------------------------------
if st.button("Predecir riesgo"):
 
    Y_pred = modelo.predict(data_preparada)
    Y_pred_etiqueta = labelencoder.inverse_transform(Y_pred)
    data["Prediccion"] = Y_pred_etiqueta
 
    if Y_pred_etiqueta[0] == "Yes":
        st.markdown(
            """
            <div class="result result-danger">
                <span class="label">Resultado</span>
                <span class="big">Riesgo de ataque al corazón</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result result-safe">
                <span class="label">Resultado</span>
                <span class="big">Bajo riesgo de ataque al corazón</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
    st.markdown('<div class="section-title" style="margin-top:22px;">Resumen</div>', unsafe_allow_html=True)
    st.dataframe(data, use_container_width=True)
 