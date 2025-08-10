import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import joblib

cols_base = ['emo_enojo','emo_asco','emo_miedo','emo_feliz','emo_triste','emo_sorpresa','emo_neutro']

def cargar_datos(ruta="features_deepface.csv"):
    df = pd.read_csv(ruta)
    for c in cols_base:
        if c not in df.columns: df[c] = 0.0
    df[cols_base] = df[cols_base].fillna(0.0).clip(0.0, 1.0)
    if "dx6_pred" not in df.columns:
        raise SystemExit("Falta la columna dx6_pred. Ejecuta extrae_emociones.py primero.")
    df = df.dropna(subset=["dx6_pred"]).copy()
    df = df[(df[cols_base].sum(axis=1) > 0)].copy()
    return df

def agregar_combinaciones(df):
    df = df.copy()
    df["miedo_sorpresa"] = df["emo_miedo"] * df["emo_sorpresa"]     # ansiedad
    df["enojo_asco"]     = df["emo_enojo"] * df["emo_asco"]         # estrés
    df["triste_miedo"]   = df["emo_triste"] * df["emo_miedo"]       # depresión ansiosa
    df["feliz_sorpresa"] = df["emo_feliz"] * df["emo_sorpresa"]     # feliz eufórico
    df["tension"]        = df["emo_enojo"] + df["emo_asco"] + df["emo_miedo"]
    df["valencia"]       = df["emo_feliz"] - df["emo_triste"]
    df["activacion"]     = df["emo_sorpresa"] + df["emo_miedo"] + df["emo_enojo"] - df["emo_neutro"]
    df["neutro_alto"]    = (df["emo_neutro"] >= 0.75).astype(float)
    cols_extra = ["miedo_sorpresa","enojo_asco","triste_miedo","feliz_sorpresa","tension","valencia","activacion","neutro_alto"]
    return df, cols_extra

def balancear(X, y):
    ros = RandomOverSampler(random_state=42)
    Xb, yb = ros.fit_resample(X, y)
    return Xb, yb

def main():
    df = cargar_datos()
    df, cols_extra = agregar_combinaciones(df)

    columnas = cols_base + cols_extra
    X = df[columnas]
    y = df["dx6_pred"].astype(str)

    print("Distribución original:", dict(y.value_counts()))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    X_tr_b, y_tr_b = balancear(X_tr, y_tr)
    print("Distribución balanceada (train):", dict(pd.Series(y_tr_b).value_counts()))

    clases = sorted(y.unique())
    pesos = compute_class_weight(class_weight="balanced", classes=np.array(clases), y=y)
    cw = {c:w for c,w in zip(clases, pesos)}

    modelo = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight=cw,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_tr_b, y_tr_b)
    y_p = modelo.predict(X_te)

    print("\nClases:", clases)
    print("\nMatriz de confusión (orden de clases):", clases)
    print(confusion_matrix(y_te, y_p, labels=clases))
    print("\nReporte:\n", classification_report(y_te, y_p, labels=clases))

    joblib.dump({"modelo": modelo, "columnas": columnas, "clases": clases}, "modelo_dx6.pkl")
    print("✅ Guardado: modelo_dx6.pkl")

if __name__ == "__main__":
    main()