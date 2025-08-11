# reco_mlp.py  (versión final, todo-en-uno)
# -------------------------------------------------------------
# Qué hace
# - Entrena un MLP que mapea 6 scores (Ansiedad, Depresión, Enojado, Estrés, Feliz, Neutral)
#   -> severidad (bajo/medio/alto) por dimensión -> recomendaciones (reglas PRO).
# - Predice recomendaciones desde un JSON con tus scores.
# - Genera datasets tabulares grandes (CSV/Parquet/JSONL) a partir de las reglas.
#
# Requisitos:
#   pip install torch numpy
#   # opcional (solo si usas from_csv o quieres Parquet): pip install pandas pyarrow
#
# Comandos:
#   Entrenar (sintético gamma):
#       python reco_mlp.py train --data synthetic_gamma --epochs 10
#   Entrenar (dirichlet):
#       python reco_mlp.py train --data dirichlet --alpha 0.6
#   Entrenar (mezcla con picos dominantes):
#       python reco_mlp.py train --data mixture_peaks --peak-p 0.6
#   Entrenar con tus datos:
#       python reco_mlp.py train --data from_csv --csv ./mis_scores.csv
#
#   Predecir desde scores.json (detallado):
#       python reco_mlp.py predict --json ./scores.json
#   Predecir (una línea simple):
#       python reco_mlp.py predict --json ./scores.json --simple
#
#   Generar dataset tabular desde reglas (sin entrenar):
#       python reco_mlp.py make-dataset --n 50000 --mode gamma --out-prefix reco_dataset
#       python reco_mlp.py make-dataset --n 50000 --mode dirichlet --alpha 0.8 --out-prefix reco_dir
#       python reco_mlp.py make-dataset --n 50000 --mode mixture_peaks --peak-p 0.6 --out-prefix reco_peaks
# -------------------------------------------------------------

import os, json, argparse, random
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# pandas/pyarrow son opcionales (solo si usas from_csv o quieres Parquet)
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import pyarrow  # noqa
except Exception:
    pyarrow = None  # solo afecta a to_parquet

# -----------------------
# Taxonomía y reglas
# -----------------------
DIM6 = ["Ansiedad","Depresión","Enojado","Estrés","Feliz","Neutral"]
SEV = ["bajo","medio","alto"]

PRO: Dict[str, Dict[str, List[str]]] = {
    "Ansiedad": {
        "bajo": [
            "Respiración 4-7-8 (3 ciclos) dos veces al día.",
            "Grounding 5-4-3-2-1 (3–5 min) cuando notes nervios.",
            "Camina 10–15 min a paso cómodo; nota tu respiración.",
            "Reduce cafeína después de las 14:00 y evita energéticos.",
            "Agenda ‘tiempo de preocupación’ 10 min y pospón rumiaciones.",
            "Higiene del sueño: misma hora para dormir/levantarte.",
            "Relajación muscular progresiva corta (5 min)."
        ],
        "medio": [
            "Registro de pensamientos (situación-pensamiento-evidencia-alternativa).",
            "Exposición gradual: lista 5 situaciones y empieza por la más fácil.",
            "Meditación guiada 10 min/día (temporizador o app).",
            "Actividad física 30 min, 3–4 días/semana.",
            "Limita ‘doomscrolling’: ventana de noticias 15–20 min/día.",
            "Plan de apoyo: habla con 1 persona de confianza esta semana."
        ],
        "alto": [
            "Agenda evaluación con profesional de salud mental.",
            "Aprende respiración diafragmática para ataques de pánico.",
            "Si hay ideas de autolesión, busca ayuda inmediata/líneas de crisis."
        ]
    },
    "Depresión": {
        "bajo": [
            "Activación conductual: 1 tarea muy pequeña hoy (≤10 min).",
            "Luz solar 20–30 min por la mañana.",
            "Rutina de sueño regular y desayuno sencillo.",
            "Contacta a un amigo/familiar (mensaje o llamada corta).",
            "Registro de gratitud: 3 cosas al final del día."
        ],
        "medio": [
            "Plan semanal con puntaje placer/logro (0–10) por actividad.",
            "Divide tareas en pasos de 5–10 min y marca progreso.",
            "Reduce alcohol; prioriza comidas regulares e hidratación.",
            "Explora psicoterapia basada en evidencia (BA/TCC).",
            "Define 1 meta SMART simple para esta semana."
        ],
        "alto": [
            "Evaluación clínica prioritaria para descartar depresión mayor.",
            "Si hay desesperanza intensa o ideas de daño, acude a urgencias o línea de crisis.",
            "Elabora plan de seguridad con señales de alerta y contactos."
        ]
    },
    "Enojado": {
        "bajo": [
            "Regla de los 90 s: espera y respira antes de responder.",
            "Técnica STOP (Stop-Toma aire-Observa-Prosigue).",
            "Cuenta hasta 10 exhalando lento; luego contesta.",
            "Sal a caminar 5–10 min para descargar activación.",
            "Identifica disparadores y anótalos en el día.",
            "Usa mensajes en primera persona: “Yo siento… cuando…”."
        ],
        "medio": [
            "Acordar ‘tiempo fuera’ de 20 min en discusiones intensas.",
            "Practica solución de problemas: define, ideas, elige, prueba.",
            "Detecta señales corporales (mandíbula tensa, puños) y suelta hombros.",
            "Revisa expectativas y reevaluación cognitiva del ‘debería’.",
            "Considera programa/grupo de manejo de ira."
        ],
        "alto": [
            "Si hay riesgo de agresión o estallidos frecuentes, busca ayuda profesional.",
            "Evita decisiones importantes en el pico de la emoción; reevalúa en frío."
        ]
    },
    "Estrés": {
        "bajo": [
            "Pomodoro 25/5 con 3 prioridades del día.",
            "Regla de los 2 minutos: haz ahora lo muy rápido.",
            "Micro-pausas cada 60–90 min: estira cuello/hombros.",
            "Respiración en caja 4-4-4-4 por 2–3 min.",
            "Hidrátate y organiza tu espacio de trabajo 5 min."
        ],
        "medio": [
            "Auditoría de tiempo: registra 1 día y elimina multitarea.",
            "Bloques de concentración sin notificaciones (30–45 min).",
            "Define límites: escribe 1 guion para decir ‘no’ o renegociar.",
            "Revisa carga con tu líder y delega si es posible.",
            "Prioriza sueño 7–9 h; rutina para desconexión digital."
        ],
        "alto": [
            "Evalúa señales de burnout (agotamiento, cinismo, eficacia ↓).",
            "Consulta con profesional; ajusta expectativas/recursos laborales.",
            "Toma tiempo de recuperación planificado (días/horas)."
        ]
    },
    "Feliz": {
        "bajo": [
            "Saborea un momento agradable 60–90 s (qué ves/oyes/sientes).",
            "Comparte una buena noticia con alguien cercano.",
            "Acto de amabilidad breve hoy (mensaje o ayuda pequeña).",
            "Mantén sueño/ejercicio; no descuides rutinas básicas."
        ],
        "medio": [
            "Define 1 meta de acercamiento (SMART) para esta semana.",
            "Planifica una actividad significativa con otra persona.",
            "Registra logros del día (3 ítems) y qué los facilitó.",
            "Considera voluntariado o contribuir a alguien más."
        ],
        "alto": [
            "Mantén prácticas que te funcionan y planifica un reto saludable.",
            "Cuida equilibrio: descanso, relaciones y propósito."
        ]
    },
    "Neutral": {
        "bajo": [
            "Escaneo corporal 3 min y 3 respiraciones profundas.",
            "Elige la ‘siguiente acción física’ de 5 min y hazla.",
            "Bebe agua y camina 5 min para activar energía."
        ],
        "medio": [
            "Programa 20–30 min de actividad con sentido (música, lectura, hobby).",
            "Contacta a alguien y agenda un plan breve.",
            "Ordena un área pequeña (cajón/mesa) por 10 min."
        ],
        "alto": [
            "Define 3 objetivos semanales con revisión el domingo.",
            "Explora nuevas actividades y registra cómo te sientes."
        ]
    }
}


# -----------------------
# Utilidades
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def normalize(v: np.ndarray) -> np.ndarray:
    s = v.sum()
    if s <= 1e-12:
        out = np.zeros_like(v); out[-1] = 1.0
        return out
    return v / s

def score_to_sev_idx(x: float) -> int:
    if x < 0.33: return 0
    if x < 0.66: return 1
    return 2

def sev_idx_to_name(i: int) -> str:
    return SEV[int(i)]

# -----------------------
# Datasets
# -----------------------
class BaseScoresDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.int64)
    def __len__(self) -> int: return self.X.shape[0]
    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[i], self.Y[i]

def _label_vector(x: np.ndarray, noise_std: float) -> List[int]:
    y = []
    for v in x:
        v2 = float(np.clip(v + np.random.normal(0, noise_std), 0, 1)) if noise_std > 0 else float(v)
        y.append(score_to_sev_idx(v2))
    return y

def build_synthetic_gamma(n=100_000, noise_std=0.05, shape=0.7):
    X, Y = [], []
    for _ in range(n):
        raw = np.random.gamma(shape=shape, scale=1.0, size=6).astype(np.float32)
        x = normalize(raw)
        X.append(x); Y.append(_label_vector(x, noise_std))
    return np.stack(X), np.stack(Y)

def build_dirichlet(n=100_000, alpha=0.7, noise_std=0.04):
    X, Y = [], []
    alphas = np.ones(6, dtype=np.float32) * float(alpha)
    for _ in range(n):
        x = np.random.dirichlet(alphas).astype(np.float32)
        X.append(x); Y.append(_label_vector(x, noise_std))
    return np.stack(X), np.stack(Y)

def build_mixture_peaks(n=100_000, peak_p=0.6, alpha_rest=0.4, noise_std=0.05):
    """
    Con prob peak_p elegimos una dimensión 'pico' y le damos masa principal.
    El resto se reparte con Dirichlet(alpha_rest).
    """
    X, Y = [], []
    for _ in range(n):
        if np.random.rand() < peak_p:
            k = np.random.randint(0, 6)
            rest = np.random.dirichlet(np.ones(5)*alpha_rest)
            x = np.zeros(6, dtype=np.float32)
            x[k] = float(np.random.uniform(0.55, 0.9))
            others = (1.0 - x[k]) * rest
            x[np.arange(6) != k] = others
        else:
            x = np.random.dirichlet(np.ones(6)*0.7).astype(np.float32)
        X.append(x); Y.append(_label_vector(x, noise_std))
    return np.stack(X), np.stack(Y)

def build_from_csv(path: str, has_sev=False, noise_std=0.0):
    """
    CSV o JSONL con columnas: Ansiedad, Depresión, Enojado, Estrés, Feliz, Neutral
    Opcional: columnas sev_{dim} con {0,1,2}. Si no hay, se autoetiqueta con umbrales.
    """
    if pd is None:
        raise RuntimeError("Pandas no está instalado. `pip install pandas` para usar from_csv.")
    if path.lower().endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)

    xs = []
    for _, row in df.iterrows():
        v = np.array([float(row.get(d, 0.0)) for d in DIM6], dtype=np.float32)
        xs.append(normalize(v))
    X = np.stack(xs)

    if has_sev:
        ys = []
        for _, row in df.iterrows():
            ys.append([int(row.get(f"sev_{d}", score_to_sev_idx(float(row.get(d,0.0))))) for d in DIM6])
        Y = np.array(ys, dtype=np.int64)
    else:
        Y = np.array([_label_vector(x, noise_std) for x in X], dtype=np.int64)

    return X, Y

# -----------------------
# Modelo MLP
# -----------------------
class RecoMLP(nn.Module):
    def __init__(self, hidden=64, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in range(6)])
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        z = self.backbone(x)
        return [h(z) for h in self.heads]

# -----------------------
# Entrenamiento / Validación
# -----------------------
def make_datasets(args) -> Tuple[BaseScoresDataset, BaseScoresDataset]:
    if args.data == "synthetic_gamma":
        Xtr,Ytr = build_synthetic_gamma(n=args.n_train, noise_std=args.noise_std, shape=args.gamma_shape)
        Xva,Yva = build_synthetic_gamma(n=args.n_val,   noise_std=args.noise_std, shape=args.gamma_shape)
    elif args.data == "dirichlet":
        Xtr,Ytr = build_dirichlet(n=args.n_train, alpha=args.alpha, noise_std=args.noise_std)
        Xva,Yva = build_dirichlet(n=args.n_val,   alpha=args.alpha, noise_std=args.noise_std)
    elif args.data == "mixture_peaks":
        Xtr,Ytr = build_mixture_peaks(n=args.n_train, peak_p=args.peak_p, alpha_rest=args.alpha_rest, noise_std=args.noise_std)
        Xva,Yva = build_mixture_peaks(n=args.n_val,   peak_p=args.peak_p, alpha_rest=args.alpha_rest, noise_std=args.noise_std)
    elif args.data == "from_csv":
        if not args.csv:
            raise ValueError("--csv es obligatorio con data=from_csv")
        X, Y = build_from_csv(args.csv, has_sev=args.csv_has_sev, noise_std=args.noise_std)
        n = X.shape[0]; cut = max(1, int(n*0.9))
        Xtr,Ytr,Xva,Yva = X[:cut],Y[:cut],X[cut:],Y[cut:]
    else:
        raise ValueError(f"data desconocido: {args.data}")

    if args.save_ds:
        os.makedirs(args.save_ds, exist_ok=True)
        np.save(os.path.join(args.save_ds,"X_train.npy"), Xtr)
        np.save(os.path.join(args.save_ds,"Y_train.npy"), Ytr)
        np.save(os.path.join(args.save_ds,"X_val.npy"),   Xva)
        np.save(os.path.join(args.save_ds,"Y_val.npy"),   Yva)
        print(f"💾 Dataset guardado en: {args.save_ds}")

    return BaseScoresDataset(Xtr,Ytr), BaseScoresDataset(Xva,Yva)

def epoch_eval(model: nn.Module, dl_val: DataLoader, device: str) -> float:
    model.eval()
    acc_sum, cnt = 0.0, 0
    with torch.no_grad():
        for xb, yb in dl_val:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            acc = 0.0
            for d in range(6):
                pred = logits[d].argmax(1)
                acc += (pred == yb[:,d]).float().mean().item()
            acc_sum += acc/6.0
            cnt += 1
    return acc_sum / max(cnt, 1)

def train_model(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_train, ds_val = make_datasets(args)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False)

    model = RecoMLP(hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()

    best = -1.0
    patience = args.patience
    wait = 0

    for ep in range(1, args.epochs+1):
        model.train()
        tot = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = sum(loss_fn(logits[d], yb[:,d]) for d in range(6))
            loss.backward()
            opt.step()
            tot += loss.item()
        tr_loss = tot / max(1,len(dl_train))

        val_acc = epoch_eval(model, dl_val, device)
        print(f"[Epoch {ep:02d}] loss={tr_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  ✔ mejor modelo guardado en {args.out}")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹️ Early stopping (paciencia {patience})")
                break

    print(f"✅ Entrenado. Mejor val_acc={best:.4f}")

def load_model(path="reco_mlp.pt", device: Optional[str]=None) -> Tuple[nn.Module, str]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = RecoMLP(hidden=64, dropout=0.0).to(device)
    # Carga de *solo pesos* para evitar pickle arbitrario
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device

# -----------------------
# Inferencia
# -----------------------
def predict_from_scores(model: nn.Module, device: str, scores: Dict[str, Any], simple=False):
    x = np.array([float(scores.get(d, 0.0)) for d in DIM6], dtype=np.float32)
    x = normalize(x)
    xb = torch.from_numpy(x[None,:]).to(device)
    with torch.no_grad():
        logits: List[torch.Tensor] = model(xb)
        sev_idx = [lg.softmax(1).argmax(1).item() for lg in logits]
    primary = DIM6[int(np.argmax(x))]
    if simple:
        s = SEV[sev_idx[DIM6.index(primary)]]
        return f"{primary} ({s}) — {PRO[primary][s][0]}"
    out = {}
    for i,d in enumerate(DIM6):
        s = SEV[sev_idx[i]]
        out[d] = {
            "score": round(float(x[i]),4),
            "severity": s,
            "acciones": PRO[d][s],
            "referir_profesional": bool(s=="alto" and d in {"Ansiedad","Depresión","Enojado","Estrés"})
        }
    return {"label_principal": primary, "detalles": out}

# -----------------------
# Generador de dataset tabular (reglas -> tabla)
# -----------------------
def sample_scores(mode: str, alpha: float, peak_p: float) -> np.ndarray:
    if mode == "dirichlet":
        return np.random.dirichlet([alpha]*6).astype(np.float32)
    elif mode == "mixture_peaks":
        if random.random() < peak_p:
            k = random.randrange(6)
            base = np.random.dirichlet([0.3]*6).astype(np.float32)
            base[k] += np.random.uniform(0.5, 1.0)
            return normalize(base)
        else:
            return np.random.dirichlet([0.7]*6).astype(np.float32)
    else:
        raw = np.random.gamma(shape=0.7, scale=1.0, size=6).astype(np.float32)
        return normalize(raw)

def build_row(idx: int, scores: np.ndarray) -> Dict[str, Any]:
    detalles = {}
    for i, dim in enumerate(DIM6):
        sev = sev_idx_to_name(score_to_sev_idx(float(scores[i])))
        detalles[dim] = {
            "score": float(round(scores[i], 6)),
            "severity": sev,
            "acciones": PRO[dim][sev]
        }
    primary = DIM6[int(np.argmax(scores))]
    row = {"id": idx, "primary_label": primary}
    for i, dim in enumerate(DIM6):
        row[f"score_{dim}"] = float(scores[i])
        row[f"sev_{dim}"]   = detalles[dim]["severity"]
        row[f"rec_{dim}"]   = detalles[dim]["acciones"][0]
    row["detalles_json"] = json.dumps(detalles, ensure_ascii=False)
    return row

def make_dataset_cli(args):
    n = args.n
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        s = sample_scores(args.mode, args.alpha, args.peak_p)
        rows.append(build_row(i, s))

    # Preferimos pandas si está disponible
    if pd is not None:
        df = pd.DataFrame(rows).astype({
            **{f"score_{d}": "float32" for d in DIM6},
            **{f"sev_{d}": "string" for d in DIM6},
            "primary_label": "string"
        })
        csv_path = f"{args.out_prefix}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"✅ CSV: {csv_path}")

        if pyarrow is not None:
            parquet_path = f"{args.out_prefix}.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"✅ Parquet: {parquet_path}")

        jsonl_path = f"{args.out_prefix}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                f.write(json.dumps({
                    "id": int(r["id"]),
                    "primary_label": r["primary_label"],
                    "scores": {d: float(r[f"score_{d}"]) for d in DIM6},
                    "detalles": json.loads(r["detalles_json"])
                }, ensure_ascii=False) + "\n")
        print(f"✅ JSONL: {jsonl_path}")
    else:
        # fallback sin pandas
        jsonl_path = f"{args.out_prefix}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"⚠️ pandas no está instalado. Se generó solo JSONL: {jsonl_path}")

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Entrenamiento
    t = sub.add_parser("train", help="Entrena el MLP con distintos datasets")
    t.add_argument("--data", choices=["synthetic_gamma","dirichlet","mixture_peaks","from_csv"], default="synthetic_gamma")
    t.add_argument("--n-train", type=int, default=120000)
    t.add_argument("--n-val",   type=int, default=12000)
    t.add_argument("--noise-std", type=float, default=0.05)
    t.add_argument("--gamma-shape", type=float, default=0.7)
    t.add_argument("--alpha", type=float, default=0.7)          # dirichlet
    t.add_argument("--peak-p", type=float, default=0.6)         # mixture_peaks
    t.add_argument("--alpha-rest", type=float, default=0.4)     # mixture_peaks
    t.add_argument("--csv", type=str, default=None)             # from_csv
    t.add_argument("--csv-has-sev", action="store_true")        # from_csv
    t.add_argument("--save-ds", type=str, default=None)         # guardar npy del dataset
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=0.0)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--out", type=str, default="reco_mlp.pt")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--patience", type=int, default=3, help="early stopping (épocas sin mejorar)")

    # Predicción
    p = sub.add_parser("predict", help="Predice recomendaciones desde scores.json")
    p.add_argument("--json", required=True)
    p.add_argument("--model", type=str, default="reco_mlp.pt")
    p.add_argument("--simple", action="store_true")

    # Generador de dataset tabular
    m = sub.add_parser("make-dataset", help="Genera dataset tabular a partir de reglas")
    m.add_argument("--n", type=int, default=50000)
    m.add_argument("--mode", choices=["gamma","dirichlet","mixture_peaks"], default="gamma")
    m.add_argument("--alpha", type=float, default=0.8)
    m.add_argument("--peak-p", type=float, default=0.6)
    m.add_argument("--out-prefix", type=str, default="reco_dataset")

    args = ap.parse_args()

    if args.cmd == "train":
        train_model(args)
    elif args.cmd == "predict":
        model, device = load_model(args.model)
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        scores = data.get("scores", data)  # acepta {"scores": {...}} o directamente {...}
        res = predict_from_scores(model, device, scores, simple=args.simple)
        if isinstance(res, str):
            print(res)
        else:
            print(json.dumps(res, ensure_ascii=False, indent=2))
    else:  # make-dataset
        make_dataset_cli(args)

if __name__ == "__main__":
    main()
