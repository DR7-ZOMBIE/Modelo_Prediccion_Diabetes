"""
Funciones de apoyo comunes
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

# Constants
RANDOM_STATE = 42
TARGET = "Diabetes_012"

def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Filtrar prediabéticos (clase 1)
    df = df[df["Diabetes_012"] != 1]

    # Re-codificar: 0 = no diabético, 2 = diabético → 1
    df["Diabetes_012"] = (df["Diabetes_012"] == 2).astype(int)

    return df


def build_preprocess_pipeline(numeric_features: list[str]) -> ColumnTransformer:
    """
    Devuelve un ColumnTransformer que:
    • Estandariza las variables numéricas
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_features)],
        remainder="passthrough",           # el resto se deja igual (ya son 0/1 o enteros)
        verbose_feature_names_out=False
    )
    return preprocessor

def build_model(class_weight: str | dict = "balanced") -> LogisticRegression:
    """
    Regresión logística multinomial con manejo de desbalanceo
    """
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=200,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return model


def train_pipeline(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split

    # Submuestreo (undersampling manual)
    df_majority = df[df[TARGET] == 0]
    df_minority = df[df[TARGET] == 1]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    X = df_balanced.drop(columns=[TARGET])
    y = df_balanced[TARGET]

    numeric_features = X.columns.tolist()
    preprocessor = build_preprocess_pipeline(numeric_features)
    clf = build_model()

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", clf),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }

    return pipeline, metrics

def save_artifacts(pipeline, metrics, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "model.pkl")
    joblib.dump(metrics, out_dir / "metrics.pkl")
