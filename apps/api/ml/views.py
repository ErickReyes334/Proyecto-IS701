# api/ml/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .modelo_cargador import obtener_modelo
from .utils import vector_desde_bytes
from .consejos import generar_consejos  # 👈 nuevo import

# Carga/caché del modelo emocional
MODELO, COLUMNAS, CLASES = obtener_modelo()

class PrediccionImagenView(APIView):
    def post(self, request):
        if 'imagen' not in request.FILES:
            return Response({"error": "Falta el archivo 'imagen' (multipart/form-data)."},
                            status=status.HTTP_400_BAD_REQUEST)
        imagen = request.FILES['imagen']
        try:
            # 1️⃣ Predicción de emociones
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

            resultado_prediccion = {
                "label": pred,
                "scores": scores,
                "emo_features": fila
            }

            # 2️⃣ Consejos dinámicos
            k = int(request.GET.get("k", 3))
            por_clase = int(request.GET.get("por_clase", 2))
            semilla = int(request.GET.get("semilla", 42))
            consejos = generar_consejos(resultado_prediccion, k=k, por_clase=por_clase, semilla=semilla)

            # 3️⃣ Respuesta combinada
            return Response({**resultado_prediccion, "consejos": consejos})

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
