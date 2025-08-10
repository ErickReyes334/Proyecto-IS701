import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
from mapeo_dx6 import mapear_emociones_dx6_fila

carpeta = "images"
salida = "features_deepface.csv"
limite = 3000
detector = "opencv"
claves = ['angry','disgust','fear','happy','sad','surprise','neutral']
cols_es = ['emo_enojo','emo_asco','emo_miedo','emo_feliz','emo_triste','emo_sorpresa','emo_neutro']
map_keys = dict(zip(claves, cols_es))

def listar_imagenes(d):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    r = []
    for e in exts: r += glob.glob(os.path.join(d, e))
    return sorted(r)

def analizar(ruta):
    f = {"ruta": ruta}
    try:
        r = DeepFace.analyze(img_path=ruta, actions=['emotion','age','gender'],
                             detector_backend=detector, align=True, enforce_detection=False)
        if isinstance(r, list): r = r[0]
        emo = r.get("emotion") or {}
        for k, c in map_keys.items():
            v = emo.get(k)
            f[c] = (float(v)/100.0 if v and float(v) > 1.0 else float(v)) if v is not None else np.nan
        f["emocion_dominante"] = r.get("dominant_emotion")
        f["edad_estimada"] = r.get("age")
        f["genero_estimado"] = r.get("gender")
        f["confianza_rostro"] = r.get("face_confidence")
    except Exception:
        for c in cols_es: f[c] = np.nan
        f.update({"emocion_dominante": None, "edad_estimada": np.nan,
                  "genero_estimado": None, "confianza_rostro": np.nan})
    return f

def main():
    imgs = listar_imagenes(carpeta)
    if limite: imgs = imgs[:limite]
    if not imgs:
        print(f"No hay imágenes en '{carpeta}'"); return

    filas = [analizar(p) for p in tqdm(imgs, desc="DeepFace")]
    df = pd.DataFrame(filas)
    for c in cols_es:
        if c not in df.columns: df[c] = 0.0
        df[c] = df[c].fillna(0.0).clip(0.0, 1.0)

    df["dx6_pred"] = df.apply(mapear_emociones_dx6_fila, axis=1)
    df.to_csv(salida, index=False)
    print(f"Listo ✅ Guardado: {salida} (filas: {len(df)})")

if __name__ == "__main__":
    main()