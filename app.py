# === app.py ===
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

st.set_page_config(
    page_title="Clasificación de Diabetes – Clínica Endocrinológica",
    page_icon="🩺",
    layout="wide",
)

ARTIFACTS = Path("artifacts")
MODEL = joblib.load(ARTIFACTS / "model.pkl")
METRICS = joblib.load(ARTIFACTS / "metrics.pkl")
FEATURES = MODEL.named_steps["prep"].get_feature_names_out()

# Funciones de visualización interactiva

def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {auc:.3f}", mode='lines'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', mode='lines', line=dict(dash='dash')))
    fig.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR")
    return fig

def plot_pr(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, name=f"AP = {ap:.3f}", mode='lines'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], name='Referencia', mode='lines', line=dict(dash='dash')))
    fig.update_layout(title="Curva Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")
    return fig

def plot_calibration(y_true, y_proba, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Calibración'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Perfecto', mode='lines', line=dict(dash='dash')))
    fig.update_layout(title="Curva de Calibración", xaxis_title="Probabilidad media predicha", yaxis_title="Frecuencia real")
    return fig

# === INTERFAZ ===
st.title("🩺 Predicción de Diabetes\n")
st.markdown("""
Aplicación desarrollada siguiendo la metodología **CRISP-DM**.  
Modelo: *Regresión logística binaria* con balanceo de clases.  
Dataset clínico procesado.  
""")

# Tabs UI
tab_pred, tab_data, tab_metrics = st.tabs(["🔮 Predicción", "📊 Datos", "📈 Métricas"])

# ---------- Predicción ----------
with tab_pred:
    st.header("Ingresar características del paciente")
    cols = st.columns(3)
    user_input = {}
    for i, feat in enumerate(FEATURES):
        col = cols[i % 3]
        default = 25.0 if feat.lower() == "bmi" else 0.0
        user_input[feat] = col.number_input(feat, value=default)

    if st.button("Predecir estado glucémico"):
        X_new = pd.DataFrame([user_input])
        pred = MODEL.predict(X_new)[0]
        proba = MODEL.predict_proba(X_new)[0][1]  # Probabilidad de ser diabético
        status = "Diabético" if pred == 1 else "No diabético"
        st.success(f"**Resultado:** {status} (Prob: {proba:.2%})")

# ---------- Datos ----------
with tab_data:
    st.header("Exploración de datos")
    DATA_PATH = Path("data/datos_clasificacion.csv")
    df_original = pd.read_csv(DATA_PATH)

    # 1️⃣ Distribución original (3 clases)
    st.subheader("Distribución ORIGINAL de la variable objetivo")
    dist_original = df_original["Diabetes_012"].value_counts().sort_index()
    st.bar_chart(dist_original)

    # 2️⃣ Distribución tras limpieza (solo clases 0 y 2 → 0 y 1)
    df_filtered = df_original[df_original["Diabetes_012"] != 1].copy()
    df_filtered["Diabetes_012"] = (df_filtered["Diabetes_012"] == 2).astype(int)
    st.subheader("Distribución tras limpieza y recodificación")
    dist_filtered = df_filtered["Diabetes_012"].value_counts().sort_index()
    st.bar_chart(dist_filtered)

    # 3️⃣ Distribución balanceada usando el mismo método que en train_pipeline
    from sklearn.utils import resample

    df_majority = df_filtered[df_filtered["Diabetes_012"] == 0]
    df_minority = df_filtered[df_filtered["Diabetes_012"] == 1]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    st.subheader("Distribución tras balanceo (undersampling)")
    dist_balanced = df_balanced["Diabetes_012"].value_counts().sort_index()
    st.bar_chart(dist_balanced)

    st.subheader("Primeras filas del dataset limpio y balanceado")
    st.dataframe(df_balanced.head(), use_container_width=True)


# ---------- Métricas ----------
# ---------- Métricas ----------
with tab_metrics:
    st.header("Métricas del modelo")
    st.markdown("Estas métricas se han calculado **después de balancear la clase mayoritaria por submuestreo**.")
    
    st.metric("Accuracy", f"{METRICS['accuracy']:.3f}")
    st.metric("F1-macro", f"{METRICS['f1_macro']:.3f}")

    st.subheader("Matriz de confusión (post-balanceo)")
    fig, ax = plt.subplots()
    sns.heatmap(
        METRICS["conf_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No diabético (0)", "Diabético (1)"],
        yticklabels=["No diabético (0)", "Diabético (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    st.pyplot(fig)

    st.subheader("Reporte de clasificación")
    st.json(METRICS["report"])

    st.subheader("Gráficas interactivas")
    st.plotly_chart(plot_roc(METRICS["y_true"], METRICS["y_proba"]), use_container_width=True)
    st.plotly_chart(plot_pr(METRICS["y_true"], METRICS["y_proba"]), use_container_width=True)
    st.plotly_chart(plot_calibration(METRICS["y_true"], METRICS["y_proba"]), use_container_width=True)
