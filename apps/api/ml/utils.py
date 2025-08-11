from tempfile import NamedTemporaryFile
import os, pandas as pd
from deepface import DeepFace

def _n(x):
    if x is None: return 0.0
    x = float(x)
    return x/100.0 if x > 1.0 else x

def mapear_emociones_dx6_fila(f):
    enojo    = _n(f.get('emo_enojo'))
    asco     = _n(f.get('emo_asco'))
    miedo    = _n(f.get('emo_miedo'))
    feliz    = _n(f.get('emo_feliz'))
    triste   = _n(f.get('emo_triste'))
    sorpresa = _n(f.get('emo_sorpresa'))
    neutro   = _n(f.get('emo_neutro'))

    if neutro >= 0.80 and max(feliz, triste, miedo, enojo, asco, sorpresa) <= 0.35:
        return "Neutral"
    if feliz >= 0.75 and (triste + miedo + enojo) <= 0.55:
        return "Feliz"
    if enojo >= 0.80 and (enojo - feliz) >= 0.15 and enojo >= max(triste, miedo):
        return "Enojado"
    if asco >= 0.80 and feliz <= 0.40:
        return "Estrés"
    if asco >= 0.60 and enojo >= 0.50 and feliz <= 0.45:
        return "Estrés"

    if miedo >= 0.65 and sorpresa >= 0.45 and feliz <= 0.45:
        return "Ansiedad"
    if triste >= 0.65 and feliz <= 0.35:
        return "Depresión"
    if triste >= 0.55 and miedo >= 0.45 and feliz <= 0.45:
        return "Depresión"
    if enojo >= 0.60 and miedo >= 0.50 and feliz <= 0.45:
        return "Enojado"
    if enojo >= 0.60 and asco >= 0.50 and feliz <= 0.45:
        return "Estrés"
    if feliz >= 0.60 and sorpresa >= 0.50 and (triste + miedo + enojo) <= 0.70:
        return "Feliz"

    neg_altas = sum(v >= 0.55 for v in [enojo, asco, miedo, triste])
    if neg_altas >= 2 and feliz <= 0.45:
        if miedo >= 0.55:
            return "Ansiedad"
        if enojo >= 0.55 or asco >= 0.55:
            return "Estrés"
        return "Depresión"

    s_feliz  = 1.15*feliz - 0.55*triste - 0.35*miedo - 0.25*enojo
    s_dep    = 1.30*triste + 0.20*miedo + 0.10*enojo - 0.60*feliz - 0.20*asco
    s_ans    = 1.30*miedo + 0.60*sorpresa - 0.30*feliz
    s_est    = 1.10*enojo + 1.00*asco + 0.50*miedo + 0.30*triste - 0.40*feliz
    s_enojo  = 1.25*enojo + 0.50*asco - 0.40*feliz
    s_neutro = 0.85*neutro - 0.45*(enojo + triste + miedo) - 0.25*feliz

    puntajes = {
        "Feliz": s_feliz, "Depresión": s_dep, "Ansiedad": s_ans,
        "Estrés": s_est, "Enojado": s_enojo, "Neutral": s_neutro
    }
    m = max(puntajes.values())
    cand = [k for k,v in puntajes.items() if abs(v-m) < 1e-9]
    if len(cand) == 1: return cand[0]
    if "Neutral" in cand and neutro < 0.75:
        cand = [c for c in cand if c != "Neutral"]
    for p in ["Enojado","Estrés","Ansiedad","Depresión","Feliz","Neutral"]:
        if p in cand: return p

def _extrae_emociones_desde_ruta(ruta, detector):
    r = DeepFace.analyze(
        img_path=ruta,
        actions=['emotion'],
        detector_backend=detector,
        align=True,
        enforce_detection=False
    )
    if isinstance(r, list): r = r[0]
    emo = r.get("emotion") or {}
    return {
        "emo_enojo":    _n(emo.get("angry", 0.0)),
        "emo_asco":     _n(emo.get("disgust", 0.0)),
        "emo_miedo":    _n(emo.get("fear", 0.0)),
        "emo_feliz":    _n(emo.get("happy", 0.0)),
        "emo_triste":   _n(emo.get("sad", 0.0)),
        "emo_sorpresa": _n(emo.get("surprise", 0.0)),
        "emo_neutro":   _n(emo.get("neutral", 0.0)),
    }

def _agrega_combinadas(base: dict, columnas_que_pide):
    enojo, asco, miedo = base["emo_enojo"], base["emo_asco"], base["emo_miedo"]
    feliz, triste, sorpresa, neutro = base["emo_feliz"], base["emo_triste"], base["emo_sorpresa"], base["emo_neutro"]
    if "miedo_sorpresa" in columnas_que_pide: base["miedo_sorpresa"] = miedo + sorpresa
    if "enojo_asco" in columnas_que_pide:     base["enojo_asco"] = enojo + asco
    if "triste_miedo" in columnas_que_pide:   base["triste_miedo"] = triste + miedo
    if "feliz_sorpresa" in columnas_que_pide: base["feliz_sorpresa"] = feliz + sorpresa
    if "tension" in columnas_que_pide:        base["tension"] = 0.7*enojo + 0.6*asco + 0.4*miedo + 0.3*triste - 0.3*feliz
    if "valencia" in columnas_que_pide:       base["valencia"] = feliz - (triste + enojo + asco + miedo)
    if "activacion" in columnas_que_pide:     base["activacion"] = feliz + sorpresa + miedo + enojo
    if "neutro_alto" in columnas_que_pide:    base["neutro_alto"] = 1.0 if neutro >= 0.80 else 0.0

def vector_desde_bytes(img_bytes, columnas, detector="retinaface"):
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(img_bytes)
        ruta = tmp.name
    try:
        fila = _extrae_emociones_desde_ruta(ruta, detector)
        _agrega_combinadas(fila, columnas)
        for c in columnas:
            if c not in fila: fila[c] = 0.0
        X = pd.DataFrame([fila])[columnas]
        return fila, X
    finally:
        try: os.remove(ruta)
        except Exception: pass
