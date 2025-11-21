import streamlit as st
import pandas as pd
import joblib
import os

# =============================
# Configuración general
# =============================
st.set_page_config(page_title="Predicción de Riesgo de Crédito", layout="wide")

# =============================
# Estilos visuales
# =============================
st.markdown("""
<style>
.stApp { background-color: #F2D7DA; color: black; }
.stButton>button {
    background-color: #800020;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 8px 16px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# CARGA DEL MODELO
# =============================
modelo_path = "modelo_gb.pkl"
if not os.path.exists(modelo_path):
    st.error(f"No se encontró el modelo en la ruta: {modelo_path}")
    st.stop()
model = joblib.load(modelo_path)

# =============================
# COLUMNAS ESPERADAS
# =============================
expected_cols = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# =============================
# MAPEOS PARA EDUCATION Y MARRIAGE
# =============================
education_map = {
    "Universidad (1)": 1,
    "Posgrado (2)": 2,
    "Secundaria (3)": 3,
    "Otros (4)": 4,
    "Desconocido (0,5,6)": 0
}

marriage_map = {
    "Casado/a (1)": 1,
    "Soltero/a (2)": 2,
    "Otros (3)": 3,
    "Desconocido (0)": 0
}

# =============================
# CREACIÓN DE TABS
# =============================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Inicio",
    "🧩 Predicción Interactiva",
    "📂 Predicción con CSV",
    "🚀 Próximos Pasos & Ideas"
])

# -----------------------------------------------------
# TAB 1: INICIO
# -----------------------------------------------------
with tab1:
    st.image("logo.png", width=250)
    st.title("Taiwan Bank – Sistema Inteligente de Detección de Riesgo 💳")

    st.write("""
    Esta aplicación permite a los analistas del **Taiwan Bank** detectar clientes con mayor riesgo 
    de impago utilizando modelos de *machine learning*.

    Funcionalidades:
    - 🔍 Predicción manual cliente a cliente  
    - 📂 Análisis masivo mediante carga CSV con filtros para segmentación. 
    - 💡 Área de sugerencias para mejoras    
    """)

# -----------------------------------------------------
# TAB 2: PREDICCIÓN INTERACTIVA
# -----------------------------------------------------
with tab2:
    st.header("🔍 Predicción Interactiva de Riesgo")

    st.write("Introduce los datos del cliente para obtener su probabilidad de riesgo:")

    # -----------------------------
    # Inputs básicos
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        limit_bal = st.number_input("Límite de Crédito (LIMIT_BAL)", min_value=0, value=20000)
        age = st.number_input("Edad", min_value=18, max_value=100, value=35)
        sex = st.selectbox("Sexo", [1, 2])

    with col2:
        education_label = st.selectbox("Nivel de Educación", list(education_map.keys()))
        education = education_map[education_label]

        marriage_label = st.selectbox("Estado Civil", list(marriage_map.keys()))
        marriage = marriage_map[marriage_label]

        # Historial de pagos PAY_0 → PAY_3
        pay_0 = st.number_input("PAY_0", -2, 9, 0)
        pay_2 = st.number_input("PAY_2", -2, 9, 0)
        pay_3 = st.number_input("PAY_3", -2, 9, 0)

    with col3:
        pay_4 = st.number_input("PAY_4", -2, 9, 0)
        pay_5 = st.number_input("PAY_5", -2, 9, 0)
        pay_6 = st.number_input("PAY_6", -2, 9, 0)

    st.subheader("📄 Facturación y Pagos Últimos 6 Meses")

    # -----------------------------
    # BILL_AMT1-6 y PAY_AMT1-6
    # -----------------------------
    bill_inputs = {}
    pay_amt_inputs = {}
    for i in range(1, 7):
        bill_inputs[f"BILL_AMT{i}"] = st.number_input(f"BILL_AMT{i}", value=0)
        pay_amt_inputs[f"PAY_AMT{i}"] = st.number_input(f"PAY_AMT{i}", value=0)

    # -----------------------------
    # Botón de predicción
    # -----------------------------
    if st.button("🔮 Predecir riesgo"):
        # Crear DataFrame con todas las columnas requeridas
        new_data = pd.DataFrame([{
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            "PAY_0": pay_0,
            "PAY_2": pay_2,
            "PAY_3": pay_3,
            "PAY_4": pay_4,
            "PAY_5": pay_5,
            "PAY_6": pay_6,
            **bill_inputs,
            **pay_amt_inputs
        }])

        # Predicción
        pred = model.predict(new_data)[0]
        proba = model.predict_proba(new_data)[0, 1]

        st.success(f"🎯 Predicción: **{'ALTO RIESGO (1)' if pred == 1 else 'BAJO RIESGO (0)'}**")
        st.info(f"📊 Probabilidad de Impago: **{round(proba*100,2)}%**")


# -----------------------------------------------------
# TAB 3: CSV UPLOAD + FILTROS ORIGINALES
# -----------------------------------------------------
with tab3:
    st.header("📂 Predicción por Archivo CSV")

    file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file, sep=';')
        except:
            st.error("Error al leer el archivo")
            st.stop()

        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            st.error(f"Faltan columnas: {missing_cols}")
            st.stop()

        df = df[expected_cols]

        preds = model.predict(df)
        probas = model.predict_proba(df)[:, 1]

        df_result = df.copy()
        df_result["pred_riesgo"] = preds
        df_result["prob_riesgo"] = probas

        st.markdown("---")
        st.subheader("Filtros")

        sexo_options = ["Todos"] + sorted(df["SEX"].unique().tolist())
        sexo_filtro = st.selectbox("Filtrar por sexo", sexo_options)

        edad_min, edad_max = st.slider(
            "Rango de Edad",
            int(df["AGE"].min()), int(df["AGE"].max()),
            (int(df["AGE"].min()), int(df["AGE"].max()))
        )

        limite_min, limite_max = st.slider(
            "Rango de Límite de Crédito",
            int(df["LIMIT_BAL"].min()), int(df["LIMIT_BAL"].max()),
            (int(df["LIMIT_BAL"].min()), int(df["LIMIT_BAL"].max()))
        )

        riesgo_filtro = st.selectbox("Filtrar por riesgo predicho",
                                     ["Todos", "Alto (1)", "Bajo (0)"])

        df_filtered = df_result[
            (df_result["AGE"] >= edad_min) &
            (df_result["AGE"] <= edad_max) &
            (df_result["LIMIT_BAL"] >= limite_min) &
            (df_result["LIMIT_BAL"] <= limite_max)
        ]

        if sexo_filtro != "Todos":
            df_filtered = df_filtered[df_filtered["SEX"] == sexo_filtro]

        if riesgo_filtro == "Alto (1)":
            df_filtered = df_filtered[df_filtered["pred_riesgo"] == 1]
        elif riesgo_filtro == "Bajo (0)":
            df_filtered = df_filtered[df_filtered["pred_riesgo"] == 0]

        st.subheader("Resultados filtrados")
        st.dataframe(df_filtered)

        csv_export = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar resultados en CSV",
            csv_export,
            "predicciones_riesgo.csv",
            "text/csv"
        )

# -----------------------------------------------------
# TAB 4: PRÓXIMOS PASOS + SUGERENCIAS
# -----------------------------------------------------
with tab4:
    st.header("🚀 Próximos pasos")

    st.write("""
   ✨ Mejoras del modelo de riesgo de crédito

🔧 Mejorar la calidad del dataset
             
📊 Crear nuevas variables financieras
             
🎯 Mejorar la precisión general del modelo
             
⚖️ Interpretar igual de bien a clientes con y sin riesgo
             
🧠 Mejorar la interpretabilidad
             
🚀 Probar modelos más complejos
             
💼 Usar métricas alineadas al negocio
             
    """)

    st.subheader("💡 Envía tus ideas o sugerencias")
    suggestion = st.text_area("Escribe aquí tu comentario:")

    
    priority = st.selectbox("¿Qué mejora consideras más importante?", [
        "Calidad del dataset", 
        "Nuevas variables financieras", 
        "Precisión del modelo", 
        "Interpretabilidad", 
        "Probar modelos complejos", 
        "Métricas de negocio"
    ])

    if st.button("Enviar comentario"):
        if suggestion.strip() != "":
            st.success("¡Gracias! Tu comentario ha sido registrado:")
            st.write(f"💬 {suggestion}")
            st.info(f"Prioridad seleccionada: {priority}")
        else:
            st.warning("Por favor, escribe tu comentario antes de enviar.")

