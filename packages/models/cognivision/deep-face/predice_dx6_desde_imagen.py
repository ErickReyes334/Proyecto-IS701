import sys
import pandas as pd
import joblib
from deepface import DeepFace

paquete = joblib.load("modelo_dx6.pkl")
modelo = paquete["modelo"]
columnas = paquete["columnas"]  # puede incluir combinaciones

def _n(x):
    x = float(x)
    return x/100.0 if x > 1.0 else x

def vector_desde_imagen(ruta, detector="opencv"):
    r = DeepFace.analyze(img_path=ruta, actions=['emotion'],
                         detector_backend=detector, align=True, enforce_detection=False)
    if isinstance(r, list): r = r[0]
    emo = r.get("emotion") or {}

    base = {
        "emo_enojo":    _n(emo.get("angry", 0.0)),
        "emo_asco":     _n(emo.get("disgust", 0.0)),
        "emo_miedo":    _n(emo.get("fear", 0.0)),
        "emo_feliz":    _n(emo.get("happy", 0.0)),
        "emo_triste":   _n(emo.get("sad", 0.0)),
        "emo_sorpresa": _n(emo.get("surprise", 0.0)),
        "emo_neutro":   _n(emo.get("neutral", 0.0)),
    }

    comb = {}
    if "miedo_sorpresa" in columnas:
        comb["miedo_sorpresa"] = base["emo_miedo"] * base["emo_sorpresa"]
    if "enojo_asco" in columnas:
        comb["enojo_asco"] = base["emo_enojo"] * base["emo_asco"]
    if "triste_miedo" in columnas:
        comb["triste_miedo"] = base["emo_triste"] * base["emo_miedo"]
    if "feliz_sorpresa" in columnas:
        comb["feliz_sorpresa"] = base["emo_feliz"] * base["emo_sorpresa"]
    if "tension" in columnas:
        comb["tension"] = base["emo_enojo"] + base["emo_asco"] + base["emo_miedo"]
    if "valencia" in columnas:
        comb["valencia"] = base["emo_feliz"] - base["emo_triste"]
    if "activacion" in columnas:
        comb["activacion"] = base["emo_sorpresa"] + base["emo_miedo"] + base["emo_enojo"] - base["emo_neutro"]
    if "neutro_alto" in columnas:
        comb["neutro_alto"] = 1.0 if base["emo_neutro"] >= 0.75 else 0.0

    fila = {**base, **comb}
    df = pd.DataFrame([fila])

    # reordenar y rellenar por si falta algo
    for c in columnas:
        if c not in df.columns:
            df[c] = 0.0
    df = df[columnas]
    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predice_dx6_desde_imagen.py ruta_imagen.jpg [opencv|retinaface|mtcnn]")
        sys.exit(1)
    ruta = sys.argv[1]
    detector = sys.argv[2] if len(sys.argv) >= 3 else "opencv"
    X = vector_desde_imagen(ruta, detector=detector)
    pred = modelo.predict(X)[0]
    print("Predicción:", pred)

    try:
        proba = modelo.predict_proba(X)[0]
        clases = paquete.get("clases", modelo.classes_)
        top = sorted(zip(clases, proba), key=lambda x: x[1], reverse=True)[:3]
        print("Top-3:", [(c, round(p,3)) for c,p in top])
    except Exception:
        pass