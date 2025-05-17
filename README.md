# 🩺 Clasificación de Diabetes – Clínica Endocrinológica

Esta aplicación predice el estado glucémico de un paciente (diabético o no) a partir de variables clínicas y de estilo de vida. El modelo ha sido entrenado utilizando técnicas de submuestreo para equilibrar las clases y mejorar el rendimiento sobre datos desbalanceados.

<div align="center">
  <img src="https://img.shields.io/badge/streamlit-app-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/model-logistic-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/CRISP--DM-framework-green?style=for-the-badge" />
</div>

---

## 🧠 Metodología

Esta solución sigue la metodología **CRISP-DM**, abarcando todas las etapas:

1. **Comprensión del negocio**: Clasificación de riesgo glucémico en pacientes.
2. **Comprensión de los datos**: Dataset clínico con más de 250,000 registros.
3. **Preparación de los datos**: Limpieza, recodificación y balanceo de clases.
4. **Modelado**: Regresión logística binaria.
5. **Evaluación**: Métricas como Accuracy, F1 y curvas ROC/PR.
6. **Despliegue**: App interactiva con Streamlit.

---

## 📊 Dataset

El dataset proviene de registros clínicos anonimizados con las siguientes características:

- Variables de estilo de vida: IMC, actividad física, consumo de alcohol, alimentación.
- Condiciones médicas: hipertensión, colesterol, salud general, salud mental/física.
- Variables sociodemográficas: edad, sexo, educación, ingresos.

La variable objetivo es `Diabetes_012`:

| Clase original | Descripción      | Tratamiento actual |
|----------------|------------------|---------------------|
| 0              | No diabético     | 0                   |
| 1              | Prediabético     | Eliminado           |
| 2              | Diabético        | 1                   |

---

## ⚖️ Balanceo de Clases

Dado que el dataset original está altamente desbalanceado (menos del 15% son diabéticos), se aplica:

- ✅ **Submuestreo de la clase mayoritaria** (`No diabético`)
- ✅ **Entrenamiento sobre datos balanceados** para evitar sesgo

---

## 🧪 Modelo

- 🔍 **Modelo**: Regresión Logística binaria (`scikit-learn`)
- ⚙️ **Balanceo**: `resample()` de `sklearn.utils`
- 🧼 **Preprocesamiento**: `StandardScaler` + `ColumnTransformer`
- 📈 **Evaluación**:
  - Matriz de Confusión
  - Curva ROC + AUC
  - Curva Precision-Recall
  - Curva de Calibración

---

## 🚀 Aplicación Interactiva (Streamlit)

La app permite:

- 🧾 Ingresar manualmente los valores de un paciente
- 🤖 Recibir una predicción (diabético o no)
- 📈 Visualizar métricas y gráficas interactivas
- 🧹 Ver cómo se limpiaron y balancearon los datos

---

## 🗂️ Estructura del proyecto

📦 diabetes_classifier
├── data/
│ └── datos_clasificacion.csv
├── artifacts/
│ ├── model.pkl
│ └── metrics.pkl
├── src/
│ ├── training.py
│ └── utils.py
├── app.py
└── requirements.txt


---
## ⚙️ Requisitos

Instala los paquetes necesarios con:

bash
pip install -r requirements.txt
---

## 🧪 Entrenamiento del modelo

python -m src.training

## 💻 Ejecutar la app

streamlit run app.py

## ✨ Autor
Dennis Juilland – Especialista en analítica y ciberseguridad
LinkedIn • GitHub • EIA
