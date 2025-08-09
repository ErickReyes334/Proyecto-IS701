import os, glob, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuración básica
DIRECTORIO_IMAGENES = "images"
CSV_CLINICO         = "mental_health_diagnosis_treatment_.csv"
MAPA_IMAGENES       = "images_map.csv"     

SALIDA_FEATURES     = "features_deepface.csv"
SALIDA_COMBINADO    = "combined_multimodal.csv"

DETECTOR   = "opencv"  
ALINEAR    = True
FORZAR_DET = False
LIMITE_IMAGENES = 3000  

EMO_CLAVES = ['angry','disgust','fear','happy','sad','surprise','neutral']

if not os.path.isdir(DIRECTORIO_IMAGENES):
    sys.exit(f"[ERROR] No existe la carpeta: {DIRECTORIO_IMAGENES}")

try:
    from deepface import DeepFace
except Exception:
    sys.exit("[ERROR] Instala deepface:  pip install deepface")

def listar_imagenes(carpeta):
    archivos = []
    for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp"):
        archivos += glob.glob(os.path.join(carpeta, ext))
    return sorted(archivos)

def analizar_imagen(ruta):
    fila = {"ruta_imagen": ruta}
    try:
        r = DeepFace.analyze(
            img_path=ruta,
            actions=['emotion','age','gender'],
            detector_backend=DETECTOR,
            align=ALINEAR,
            enforce_detection=FORZAR_DET
        )
        if isinstance(r, list) and r: r = r[0]
        fila["confianza_rostro"] = r.get("face_confidence")
        fila["emocion_dominante"] = r.get("dominant_emotion")
        fila["edad_estimada"] = r.get("age")
        fila["genero_estimado"] = r.get("gender")
        emo = r.get("emotion", {})
        for k in EMO_CLAVES:
            v = emo.get(k)
            if v is not None and v > 1.0: v = v/100.0
            fila[f"emo_{k}"] = float(v) if v is not None else np.nan
    except Exception:
        fila.update({
            "confianza_rostro": np.nan,
            "emocion_dominante": None,
            "edad_estimada": np.nan,
            "genero_estimado": None,
            **{f"emo_{k}": np.nan for k in EMO_CLAVES}
        })
    return fila

def cargar_mapa():
    if not os.path.exists(MAPA_IMAGENES):
        return None
    df = pd.read_csv(MAPA_IMAGENES)
    cols = {c.lower().strip(): c for c in df.columns}
    pid = cols.get("patient_id") or cols.get("patient id")
    path = cols.get("image_path") or cols.get("image path")
    if not pid or not path:
        print("[AVISO] 'images_map.csv' debe tener columnas: patient_id,image_path")
        return None
    df = df.rename(columns={pid:"patient_id", path:"image_path"})
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["image_path"] = df["image_path"].astype(str).str.strip()
    return df[df["image_path"].apply(os.path.exists)].copy()

def main():
    rutas = listar_imagenes(DIRECTORIO_IMAGENES)
    if LIMITE_IMAGENES: rutas = rutas[:LIMITE_IMAGENES]
    if not rutas: sys.exit(f"[ERROR] Sin imágenes en {DIRECTORIO_IMAGENES}")

    filas = [analizar_imagen(p) for p in tqdm(rutas, desc="DeepFace")]
    df_feat = pd.DataFrame(filas)
    df_feat.to_csv(SALIDA_FEATURES, index=False)
    print(f"[OK] Guardado: {SALIDA_FEATURES} ({len(df_feat)} filas)")

    mapa = cargar_mapa()
    if mapa is None:
        print("[INFO] Sin 'images_map.csv' → no se combina con el CSV clínico.")
        return
    if not os.path.exists(CSV_CLINICO):
        print(f"[AVISO] No está {CSV_CLINICO} → no se combina.")
        return

    clin = pd.read_csv(CSV_CLINICO)
    # Detectar columna Patient ID
    cols_pid = [c for c in clin.columns if c.lower().replace("_"," ").startswith("patient")]
    col_pid = cols_pid[0] if cols_pid else "Patient ID"
    if col_pid not in clin.columns:
        clin[col_pid] = np.arange(1, len(clin)+1).astype(str)
    clin[col_pid] = clin[col_pid].astype(str).str.strip()

    df_feat = df_feat.merge(mapa, left_on="ruta_imagen", right_on="image_path", how="left")
    combinado = clin.merge(df_feat, left_on=col_pid, right_on="patient_id", how="inner")
    combinado.to_csv(SALIDA_COMBINADO, index=False)
    print(f"[OK] Guardado: {SALIDA_COMBINADO} ({len(combinado)} filas)")

if __name__ == "__main__":
    main()
