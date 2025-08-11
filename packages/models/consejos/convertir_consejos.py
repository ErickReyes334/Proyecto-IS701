import json, re, pandas as pd, numpy as np
from pathlib import Path

BASE = Path(__file__).parent
SRC  = BASE / "combined_dataset.json"   
OUT  = BASE / "consejos.csv"

# --- Heurísticas simples ---
POS = r"\b(good|great|proud|grateful|calm|okay|hope|relax|win|happy|better|improve)\b"
NEG = r"\b(sad|depress|worthless|empty|hopeless|useless|anx|panic|angry|mad|stress|overwhelmed|tired|lonely|hate)\b"
TEN = r"\b(stress|tense|pressure|deadline|panic|overwhelmed|worry|tight|burnout|insomnia|sleep\s*less)\b"
ACT = r"\b(urge|racing|can.?t|cannot|restless|panic|angry|excited|energy|energized|agitated)\b"
SUI = r"\b(suicid|kill myself|end my life|don.?t want to live)\b"

def clip(a, lo, hi): return float(np.clip(a, lo, hi))

def valencia(txt):
    pos = len(re.findall(POS, txt))
    neg = len(re.findall(NEG, txt))
    return clip((pos - neg) / 10.0, -1.0, 1.0)

def tension(txt):
    k = len(re.findall(TEN, txt))
    # base 0.2 + 0.15 por palabra clave, tope 1.5
    return clip(0.2 + 0.15*k, 0.0, 1.5)

def activacion(txt):
    k = len(re.findall(ACT, txt))
    # base 0.2 + 0.2 por palabra clave, tope 2.0
    return clip(0.2 + 0.2*k, 0.0, 2.0)

def detectar_label(ctx: str):
    t = ctx.lower()
    if re.search(SUI, t):                     return "Depresión"  # crítico
    if re.search(r"\b(happy|grateful|proud|celebrat|better|hope)\b", t): return "Feliz"
    if re.search(r"\b(panic|anxious|worry|nervous|racing)\b", t):        return "Ansiedad"
    if re.search(r"\b(stress|overwhelmed|pressure|burnout)\b", t):       return "Estrés"
    if re.search(r"\b(depress|sad|empty|hopeless|worthless)\b", t):      return "Depresión"
    if re.search(r"\b(angry|mad|rage|irritat|frustrat|hate)\b", t):      return "Enojado"
    return "Neutral"

def map_consejo(label: str, ctx: str):
    t = ctx.lower()
    if re.search(SUI, t): return "Apoyo"
    return {
        "Feliz":     "Celebracion",
        "Ansiedad":  "Respiracion",
        "Estrés":    "Pausa",
        "Depresión": "Apoyo",
        "Enojado":   "Desahogo",
        "Neutral":   "Chequeo"
    }.get(label, "Chequeo")

def main():
    if not SRC.exists():
        raise SystemExit(f"No encuentro {SRC}. Pon tu JSON ahí o ajusta la ruta en el script.")

    data = json.loads(SRC.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    data = v
                    break
        if not isinstance(data, list):
            raise SystemExit("El JSON no es una lista de objetos. Revisa el formato.")

    filas = []
    for item in data:
        ctx = str(item.get("Context", "")).strip()
        if not ctx:
            continue
        lab = detectar_label(ctx)
        v   = valencia(ctx)
        ten = tension(ctx)
        act = activacion(ctx)
        cls = map_consejo(lab, ctx)
        filas.append({
            "label": lab,
            "valencia": v,
            "tension": ten,
            "activacion": act,
            "consejo_clase": cls,
            "texto": ctx[:1000],     # guarda algo del contexto (corto)
            "respuesta_demo": str(item.get("Response",""))[:1000]  # opcional, por si luego lo usas
        })

    if not filas:
        raise SystemExit("No se pudieron extraer filas del JSON.")

    df = pd.DataFrame(filas)
    # limpieza ligera (opcional)
    df = df.drop_duplicates(subset=["texto","consejo_clase"]).reset_index(drop=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"✅ Generado {OUT} con {len(df)} filas")

if __name__ == "__main__":
    main()
