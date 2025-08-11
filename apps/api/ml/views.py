from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# --- hack path ---
import os, sys
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# --- fin ---

from .modelo_cargador import obtener_modelo
from .utils import vector_desde_bytes

from packages.models.cognivision.recomendacion.reco_mlp import load_model, predict_from_scores

RECO_WEIGHTS_DEFAULT = os.path.join(
    _REPO_ROOT, "packages", "models", "cognivision", "recomendacion", "reco_mlp.pt"
)
RECO_WEIGHTS = os.environ.get("RECO_MLP_WEIGHTS", RECO_WEIGHTS_DEFAULT)
RECO_MODEL, RECO_DEVICE = load_model(RECO_WEIGHTS)

def recommend_from_scores(scores: dict, simple: bool = False):
    return predict_from_scores(RECO_MODEL, RECO_DEVICE, scores, simple=simple)

# Carga/caché del modelo emocional (tu clasificador existente)
MODELO, COLUMNAS, CLASES = obtener_modelo()

class PrediccionImagenView(APIView):
    def post(self, request):
        if 'imagen' not in request.FILES:
            return Response({"error": "Falta el archivo 'imagen' (multipart/form-data)."},
                            status=status.HTTP_400_BAD_REQUEST)
        imagen = request.FILES['imagen']
        try:
            fila, X = vector_desde_bytes(
                imagen.read(),
                columnas=COLUMNAS,
                detector=request.GET.get("detector", "retinaface")
            )
            pred = MODELO.predict(X)[0]
            if hasattr(MODELO, "predict_proba"):
                proba = MODELO.predict_proba(X)[0]
                scores = {cl: float(p) for cl, p in zip(CLASES, proba)}
            else:
                scores = {cl: (1.0 if cl == pred else 0.0) for cl in CLASES}

            simple = request.GET.get("simple", "false").lower() in {"1", "true", "yes"}
            recomendaciones = recommend_from_scores(scores, simple=simple)

            return Response({
                "label": pred,
                "scores": scores,
                "emo_features": fila,
                "recomendaciones": recomendaciones
            })
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
