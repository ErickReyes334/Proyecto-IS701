# infer_and_recommend.py
import json, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============ Mapeo GoEmotions (27+neutral) -> tus 6 dimensiones ============
DIM6 = ["Ansiedad","Depresión","Enojado","Estrés","Feliz","Neutral"]

MAP_27_TO_6 = {
    # ENOJADO
    "anger":"Enojado","annoyance":"Enojado","disgust":"Enojado","disapproval":"Enojado",
    # ANSIEDAD / ESTRÉS
    "fear":"Ansiedad","nervousness":"Ansiedad","confusion":"Estrés","realization":"Estrés",
    # DEPRESIÓN
    "sadness":"Depresión","grief":"Depresión","disappointment":"Depresión","remorse":"Depresión",
    "embarrassment":"Depresión",
    # FELIZ (afecto positivo)
    "joy":"Feliz","amusement":"Feliz","gratitude":"Feliz","love":"Feliz","optimism":"Feliz",
    "pride":"Feliz","relief":"Feliz","admiration":"Feliz","excitement":"Feliz","caring":"Feliz",
    "approval":"Feliz",
    # NEUTRAL y otros
    "neutral":"Neutral","curiosity":"Neutral","desire":"Neutral","surprise":"Neutral",
    # también presentes en la taxonomía oficial
    # (lista completa en TFDS/goemotions) :contentReference[oaicite:4]{index=4}
}

# ===== Recomendaciones profesionales por severidad =====
PRO = {
    "Ansiedad":{
        "bajo":[ "Higiene del sueño y respiración diafragmática 3–5 min." ],
        "medio":[ "Grounding 5-4-3-2-1 (10 min/día) y actividad física 30 min." ],
        "alto":[ "TCC para ansiedad / evaluación clínica; reducir cafeína; exposición guiada." ]
    },
    "Depresión":{
        "bajo":[ "Activación conductual ligera y rutina de sueño/alimentación/luz." ],
        "medio":[ "Activación conductual estructurada y red de apoyo planificada." ],
        "alto":[ "Evaluación clínica prioritaria; psicoterapia basada en evidencia." ]
    },
    "Enojado":{
        "bajo":[ "Pausa consciente + respiración antes de responder." ],
        "medio":[ "Técnica STOP y comunicación asertiva con reestructuración cognitiva." ],
        "alto":[ "Entrenamiento formal en manejo de ira / apoyo profesional si recurrente." ]
    },
    "Estrés":{
        "bajo":[ "Pomodoro 25/5 y priorización de 3 tareas." ],
        "medio":[ "Relajación muscular o mindfulness 10–15 min; revisar carga laboral." ],
        "alto":[ "Plan integral de estrés; revisar expectativas; evaluar burnout." ]
    },
    "Feliz":{
        "bajo":[ "Gratitud diaria y actividad placentera breve." ],
        "medio":[ "Consolidar hábitos protectores; compartir logros." ],
        "alto":[ "Mantener rutinas y actividades con propósito." ]
    },
    "Neutral":{
        "bajo":[ "Chequeo corporal y respiración lenta 3 min." ],
        "medio":[ "Planificar actividad significativa 20–30 min." ],
        "alto":[ "Programa de objetivos con seguimiento semanal." ]
    }
}

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def ge27_to_dim6_probs(ge_probs, ge_label_names):
    # ge_probs: array de 27+neutral probabilidades (sigmoid)
    counts = {k:0.0 for k in DIM6}
    for i, p in enumerate(ge_probs):
        name = ge_label_names[i]
        target = MAP_27_TO_6.get(name)
        if target:
            counts[target] += float(p)
    vec = np.array([counts[d] for d in DIM6], dtype=float)
    if vec.sum() <= 1e-9:
        vec[-1] = 1.0  # neutral
    vec = vec / vec.sum()
    return {d: float(v) for d, v in zip(DIM6, vec)}

def severity(score):
    if score < 0.33: return "bajo"
    if score < 0.66: return "medio"
    return "alto"

def fuse(text_scores, signal_scores, alpha=0.5):
    t = np.array([text_scores[d] for d in DIM6], dtype=float)
    s = np.array([signal_scores[d] for d in DIM6], dtype=float)
    t = t / max(t.sum(), 1e-9)
    s = s / max(s.sum(), 1e-9)
    f = alpha*t + (1-alpha)*s
    f = f / f.sum()
    return {d: float(v) for d, v in zip(DIM6, f)}

class Recommender:
    def __init__(self, model_dir="./goemo_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        with open(f"{model_dir}/labels.txt","r",encoding="utf-8") as f:
            self.ge_labels = [x.strip() for x in f if x.strip()]

    @torch.no_grad()
    def text_to_dim6(self, text, threshold=0.30):
        tokens = self.tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
        logits = self.model(**tokens).logits.squeeze().cpu().numpy()
        probs = 1/(1+np.exp(-logits))  # sigmoid por etiqueta
        # si quieres "activas", umbral: (probs>threshold)
        return ge27_to_dim6_probs(probs, self.ge_labels)

    def recommend(self, fused_scores):
        out = {}
        for d in DIM6:
            sev = severity(fused_scores[d])
            out[d] = {
                "score": round(fused_scores[d], 4),
                "severity": sev,
                "do_now": PRO[d][sev],
                "refer_to_professional": bool(sev=="alto" and d in {"Ansiedad","Depresión","Enojado","Estrés"})
            }
        primary = max(fused_scores.items(), key=lambda kv: kv[1])[0]
        return {"primary_label": primary, "per_dimension": out}

if __name__ == "__main__":
    # === ejemplo de uso ===
    # 1) texto (si no tienes texto, pon una cadena vacía y alpha bajo)
    text = "I feel overwhelmed and my heart is racing. I can't focus at work."
    # 2) tus scores de señales/emo_features ya calculados
    signal_scores = {
        "Ansiedad": 0.03594205679201842,
        "Depresión": 0.1495699296882287,
        "Enojado": 0.5436063875233454,
        "Estrés": 0.25948993707272433,
        "Feliz": 0.00889168892368311,
        "Neutral": 0.0025
    }

    rec = Recommender("./goemo_model")
    text_scores = rec.text_to_dim6(text)  # distribución 6D desde texto
    fused = fuse(text_scores, signal_scores, alpha=0.5)  # mezcla texto/señal
    result = rec.recommend(fused)
    print(json.dumps({
        "text_scores": text_scores,
        "signal_scores": signal_scores,
        "fused_scores": fused,
        "result": result
    }, ensure_ascii=False, indent=2))
