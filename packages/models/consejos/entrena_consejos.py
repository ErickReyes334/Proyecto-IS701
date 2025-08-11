from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

BASE = Path(__file__).parent
CSV  = BASE / "consejos.csv"
PKL  = BASE / "modelo_consejos.pkl"

TARGET_COL = "consejo_clase"
CAT_COLS   = ["label"]
NUM_BASE   = ["valencia", "tension", "activacion"]

NUM_SOSPECHOSAS = {
    "miedo_sorpresa", "enojo_asco", "triste_miedo", "feliz_sorpresa",
    "emo_enojo","emo_asco","emo_miedo","emo_feliz","emo_triste","emo_sorpresa","emo_neutro",
    "tension_calc","valencia_calc","activacion_calc"
}

def columnas_numericas_extras(df: pd.DataFrame):
    extras = []
    extras += [c for c in df.columns if c.startswith("emo_")]
    extras += [c for c in NUM_SOSPECHOSAS if c in df.columns]
    extras = list(dict.fromkeys(extras))
    extras = [c for c in extras if pd.api.types.is_numeric_dtype(df[c])]
    extras = [c for c in extras if c not in set([TARGET_COL] + CAT_COLS + NUM_BASE)]
    return extras

def main():
    if not CSV.exists():
        raise SystemExit(f"Falta {CSV}. Genera primero consejos.csv.")

    df = pd.read_csv(CSV)
    df = df.dropna(subset=[TARGET_COL] + CAT_COLS).copy()

    num_extras = columnas_numericas_extras(df)
    num = NUM_BASE + num_extras

    for c in num:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[CAT_COLS + num]
    y = df[TARGET_COL].astype(str)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pre = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("sc",  StandardScaler(), num),
    ])

    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=300))])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = [
        {   # Logistic Regression
            "clf": [LogisticRegression(max_iter=800, class_weight="balanced", solver="lbfgs")],
            "clf__C": [0.5, 1.0, 2.0]
        },
        {   # Random Forest
            "clf": [RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)],
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_leaf": [1, 2]
        }
    ]

    gs = GridSearchCV(
        pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True, verbose=0
    )
    gs.fit(Xtr, ytr)

    mejor = gs.best_estimator_
    print("Mejor modelo:", type(mejor.named_steps["clf"]).__name__)
    print("Mejores params:", gs.best_params_)

    # ===== Calibración (compatible con versiones nuevas y antiguas) =====
    base_clf = mejor.named_steps["clf"]
    # Intento con API nueva (estimator=...)
    try:
        cal = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
    except TypeError:
        # Fallback para scikit-learn antiguos (base_estimator=...)
        cal = CalibratedClassifierCV(base_estimator=base_clf, method="sigmoid", cv=3)

    calibrado = Pipeline([
        ("pre", mejor.named_steps["pre"]),
        ("cal", cal)
    ])
    calibrado.fit(Xtr, ytr)
    # ================================================================

    yp = calibrado.predict(Xte)
    print("\nReporte de validación:")
    print(classification_report(yte, yp, digits=3))

    plantillas = {
        "Respiracion": [
            "Pausa 60s: inhala 4, retén 2, exhala 6 (x5).",
            "Prueba 4-7-8 por 4 ciclos."
        ],
        "Pausa": [
            "Camina 5–10 min, agua y vuelve.",
            "Estira cuello/hombros 2 min y respira lento."
        ],
        "Desahogo": [
            "Escribe 2 min sin filtro y rompe la hoja.",
            "Cuenta 5→1 y cambia de contexto 3 min."
        ],
        "Celebracion": [
            "Anota 1 logro de hoy y compártelo.",
            "3 cosas que agradeces ahora mismo."
        ],
        "Apoyo": [
            "Contacta a alguien de confianza y cuéntale cómo te sientes.",
            "Ducha tibia + 10 min de sol + vaso de agua."
        ],
        "Chequeo": [
            "Chequeo rápido: energía/estrés (0–10). ¿Qué necesitas ahora?",
            "Postura recta + 5 respiraciones lentas."
        ]
    }

    joblib.dump({
        "modelo": calibrado,
        "columnas_cat": CAT_COLS,
        "columnas_num": num,
        "plantillas": plantillas,
        "meta": {
            "num_extras_detectadas": num_extras,
            "mejor_modelo_base": type(base_clf).__name__,
            "params_mejor_base": gs.best_params_,
        }
    }, PKL)

    print(f"✅ Guardado: {PKL}")
    print(f"ℹ️ Numéricas usadas: {num}")

if __name__ == "__main__":
    main()
