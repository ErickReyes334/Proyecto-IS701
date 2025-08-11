from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .modelo_cargador import obtener_modelo
from .utils import vector_desde_bytes

# Carga/caché del modelo al importar
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
                scores = {}
            return Response({"label": pred, "scores": scores, "emo_features": fila})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
