import sys, json, argparse, random
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE = Path(__file__).parent
PKL  = BASE / "modelo_consejos.pkl"
FALLBACK = "Respira hondo 30s, toma agua y date 1 minuto."

def cargar_payload(arg: str):
    p = Path(arg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(arg)

def fila_desde_payload(payload):
    label = payload.get("label")
    emo = payload.get("emo_features", {}) or {}
    v   = float(emo.get("valencia", 0.0))
    t   = float(emo.get("tension", 0.0))
    a   = float(emo.get("activacion", 0.0))
    return {"label": label, "valencia": v, "tension": t, "activacion": a}

def elegir_textos_unicos(plantillas: dict, clase: str, n: int, rng: random.Random):
  
    base = [t.strip() for t in plantillas.get(clase, []) if isinstance(t, str) and t.strip()]
   
    if len(base) < n:
        base = list(dict.fromkeys(base + [FALLBACK]*(n - len(base))))
    rng.shuffle(base)
    
    vistos, out = set(), []
    for t in base:
        if t not in vistos:
            out.append(t)
            vistos.add(t)
        if len(out) == n:
            break
    while len(out) < n:
        out.append(FALLBACK)
    return out

def main():
    ap = argparse.ArgumentParser(description="Prueba modelo de consejos (solo probabilidades del modelo).")
    ap.add_argument("input", help="JSON en línea o ruta a archivo .json")
    ap.add_argument("--k", type=int, default=3, help="Top-K clases por probabilidad")
    ap.add_argument("--por_clase", type=int, default=2, help="Número de consejos por clase")
    ap.add_argument("--semilla", type=int, default=42, help="Semilla para reproducibilidad")
    args = ap.parse_args()

    if not PKL.exists():
        print(f"Falta el modelo {PKL}. Entrena primero con: python entrena_consejos.py")
        sys.exit(1)

    rng = random.Random(args.semilla)
    np.random.seed(args.semilla)

    payload = cargar_payload(args.input)
    paquete = joblib.load(PKL)

    modelo = paquete["modelo"]
    cat = paquete["columnas_cat"]
    num = paquete["columnas_num"]
    plantillas = paquete.get("plantillas", {})

    fila = fila_desde_payload(payload)
    X = pd.DataFrame([fila])

    for c in cat + num:
        if c not in X.columns:
            X[c] = 0.0 if c in num else ""
    X = X[cat + num]

    proba = modelo.predict_proba(X)[0]
    clases = list(modelo.classes_)

    idx_sorted = np.argsort(proba)[::-1]
    top_idx = idx_sorted[:max(1, args.k)]

    sugerencias = []
    for idx in top_idx:
        clase = clases[idx]
        p = float(proba[idx])
        textos = elegir_textos_unicos(plantillas, clase, args.por_clase, rng)
        for t in textos:
            sugerencias.append({"clase": clase, "probabilidad": p, "texto": t})

    salida = {
        "entrada_label": payload.get("label"),
        "entrada_valores": fila,
        "top_probabilidades": {c: float(p) for c, p in sorted(zip(clases, proba), key=lambda kv: kv[1], reverse=True)},
        "sugerencias": sugerencias
    }
    print(json.dumps(salida, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
