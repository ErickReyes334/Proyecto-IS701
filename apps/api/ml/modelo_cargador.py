import os, subprocess, joblib

BASE_DIR = os.path.dirname(__file__)

# Carpeta raíz del proyecto 
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

# Carpeta deep-face
DEEPFACE_DIR = os.path.join(ROOT_DIR, "packages", "models", "cognivision", "deep-face")

# Ruta al modelo y al script de entrenamiento
RUTA_MODELO_PKL = os.path.join(DEEPFACE_DIR, "modelo_dx6.pkl")
RUTA_SCRIPT_ENTRENAR = os.path.join(DEEPFACE_DIR, "entrenamiento.py")

_cache = {"modelo": None, "columnas": None, "clases": None}

def cargar_o_entrenar():
    if not os.path.exists(RUTA_MODELO_PKL):
        subprocess.run(["python", RUTA_SCRIPT_ENTRENAR, "retinaface"], check=True)
    paquete = joblib.load(RUTA_MODELO_PKL)
    _cache["modelo"] = paquete["modelo"]
    _cache["columnas"] = paquete["columnas"]
    _cache["clases"] = paquete["clases"]

def obtener_modelo():
    if _cache["modelo"] is None:
        cargar_o_entrenar()
    return _cache["modelo"], _cache["columnas"], _cache["clases"]
