# api/ml/consejos.py
import random
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Ruta al modelo de consejos
PKL_PATH = Path(__file__).resolve().parents[3] / "packages" / "models" / "consejos" / "modelo_consejos.pkl"
FALLBACK = "Respira hondo 30s, toma agua y date 1 minuto."


_cache_consejos = None

def _load_model():
    global _cache_consejos
    if _cache_consejos is None:
        if not PKL_PATH.exists():
            raise FileNotFoundError(f"No se encontró el modelo de consejos en {PKL_PATH}")
        _cache_consejos = joblib.load(PKL_PATH)
    return _cache_consejos

def generar_consejos(prediccion: dict, k: int = 3, por_clase: int = 2, semilla: int = 42):
    """
    prediccion: {
        "label": str,
        "scores": {clase: prob, ...},
        "emo_features": {...}
    }
    """
    paquete = _load_model()
    modelo = paquete["modelo"]
    cat = paquete["columnas_cat"]
    num = paquete["columnas_num"]
    plantillas = paquete.get("plantillas", {})

    # Preparar fila para el modelo de consejos
    fila = {
        "label": prediccion.get("label", ""),
        "valencia": float(prediccion["emo_features"].get("valencia", 0.0)),
        "tension": float(prediccion["emo_features"].get("tension", 0.0)),
        "activacion": float(prediccion["emo_features"].get("activacion", 0.0))
    }

    X = pd.DataFrame([fila])
    for c in cat + num:
        if c not in X.columns:
            X[c] = "" if c in cat else 0.0
    X = X[cat + num]

    rng = random.Random(semilla)
    np.random.seed(semilla)

    proba = modelo.predict_proba(X)[0]
    clases = list(modelo.classes_)
    idx_sorted = np.argsort(proba)[::-1]
    top_idx = idx_sorted[:max(1, k)]

    def elegir_textos(clase, n):
        base = [t.strip() for t in plantillas.get(clase, []) if isinstance(t, str) and t.strip()]
        if len(base) < n:
            base += [FALLBACK] * (n - len(base))
        rng.shuffle(base)
        return base[:n]

    sugerencias = []
    for idx in top_idx:
        clase = clases[idx]
        p = float(proba[idx])
        for t in elegir_textos(clase, por_clase):
            sugerencias.append({
                "clase": clase,
                "probabilidad": p,
                "texto": t
            })

    return {
        "top_probabilidades": {
            c: float(p) for c, p in sorted(zip(clases, proba), key=lambda kv: kv[1], reverse=True)
        },
        "sugerencias": sugerencias
    }
