"""
Entrena el modelo, imprime métricas y guarda artefactos
Ejecute:
    python -m src.training
"""
import json, joblib
from pathlib import Path
from src.utils import load_data, train_pipeline, save_artifacts


DATA_PATH = Path(__file__).parents[1] / "data" / "datos_clasificacion.csv"
ARTIFACTS_DIR = Path(__file__).parents[1] / "artifacts"

def main():
    df = load_data(DATA_PATH)
    pipeline, metrics = train_pipeline(df)
    save_artifacts(pipeline, metrics, ARTIFACTS_DIR)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 macro:  {metrics['f1_macro']:.4f}")
    # opcional: imprimir reporte
    print(json.dumps(metrics["report"], indent=4))

if __name__ == "__main__":
    main()
