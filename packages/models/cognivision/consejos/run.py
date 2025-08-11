# run_recommend.py
import os, json, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Dimensiones y mapeos
DIM6 = ["Ansiedad","Depresión","Enojado","Estrés","Feliz","Neutral"]
MAP_27_TO_6 = {
    "anger":"Enojado","annoyance":"Enojado","disgust":"Enojado","disapproval":"Enojado",
    "fear":"Ansiedad","nervousness":"Ansiedad","confusion":"Estrés","realization":"Estrés",
    "sadness":"Depresión","grief":"Depresión","disappointment":"Depresión","remorse":"Depresión","embarrassment":"Depresión",
    "joy":"Feliz","amusement":"Feliz","gratitude":"Feliz","love":"Feliz","optimism":"Feliz","pride":"Feliz","relief":"Feliz","admiration":"Feliz","excitement":"Feliz","caring":"Feliz","approval":"Feliz",
    "neutral":"Neutral","curiosity":"Neutral","desire":"Neutral","surprise":"Neutral",
}
PRO = {
    "Ansiedad":{"bajo":["Respiración 3–5 min"],"medio":["Grounding + ejercicio"],"alto":["TCC / evaluación clínica"]},
    "Depresión":{"bajo":["Activación ligera"],"medio":["Activación estructurada"],"alto":["Evaluación clínica"]},
    "Enojado":{"bajo":["Pausa consciente"],"medio":["STOP + asertividad"],"alto":["Manejo de ira"]},
    "Estrés":{"bajo":["Pomodoro 25/5"],"medio":["Mindfulness 10–15"],"alto":["Plan integral"]},
    "Feliz":{"bajo":["Gratitud"],"medio":["Hábitos protectores"],"alto":["Rutinas con propósito"]},
    "Neutral":{"bajo":["Chequeo corporal"],"medio":["Actividad significativa"],"alto":["Objetivos con seguimiento"]},
}

def ge28_to_dim6_probs(ge_probs, ge_label_names):
    agg = {k:0.0 for k in DIM6}
    for i, p in enumerate(ge_probs):
        name = ge_label_names[i]
        target = MAP_27_TO_6.get(name)
        if target:
            agg[target] += float(p)
    vec = np.array([agg[d] for d in DIM6], dtype=float)
    if vec.sum() <= 1e-9:
        vec[-1] = 1.0
    vec = vec / vec.sum()
    return {d: float(v) for d, v in zip(DIM6, vec)}

def severity(score: float):
    if score < 0.33: return "bajo"
    if score < 0.66: return "medio"
    return "alto"

def fuse(text_scores: dict, signal_scores: dict, alpha: float = 0.5) -> dict:
    t = np.array([text_scores.get(d,0.0) for d in DIM6], dtype=float)
    s = np.array([signal_scores.get(d,0.0) for d in DIM6], dtype=float)
    t = t / max(t.sum(), 1e-9)
    s = s / max(s.sum(), 1e-9)
    f = alpha*t + (1-alpha)*s
    f = f / f.sum()
    return {d: float(v) for d, v in zip(DIM6, f)}

# === MAIN ===
if __name__ == "__main__":
    # Config
    model_dir = "./goemo_model"
    scores_file = "./scores.json"
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""  # texto opcional
    alpha = 0.5
    threshold = 0.3

    # Cargar modelo y tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    with open(os.path.join(model_dir, "labels.txt"), "r", encoding="utf-8") as f:
        ge_labels = [x.strip() for x in f if x.strip()]

    # Cargar scores del JSON
    with open(scores_file, "r", encoding="utf-8") as f:
        signal_scores = json.load(f).get("scores", {})

    # Obtener scores del texto
    if text:
        toks = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**toks).logits.squeeze().cpu().numpy()
            probs = 1/(1+np.exp(-logits))
        text_scores = ge28_to_dim6_probs(probs, ge_labels)
    else:
        text_scores = {d: (1.0 if d=="Neutral" else 0.0) for d in DIM6}

    # Fusionar y generar recomendaciones
    fused = fuse(text_scores, signal_scores, alpha=alpha)
    result = {}
    for d in DIM6:
        sev = severity(fused[d])
        result[d] = {
            "score": round(fused[d],4),
            "severity": sev,
            "acciones": PRO[d][sev],
            "referir_profesional": bool(sev=="alto" and d in {"Ansiedad","Depresión","Enojado","Estrés"})
        }
    label_principal = max(fused.items(), key=lambda kv: kv[1])[0]

    # Mostrar resultado
    print(json.dumps({
        "texto": text,
        "label_principal": label_principal,
        "detalles": result
    }, ensure_ascii=False, indent=2))
