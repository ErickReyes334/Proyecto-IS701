import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Camera, Play, Pause } from "lucide-react";
import { CameraOrFilePicker } from "@/components/face/camara";
import { usePredictImage } from "@/lib/check";
import { useState } from "react";

// ... tu código de estados/UX existente

export function CognitiveAnalyzer() {
  // ...
  const { mutate, data, isPending, error } = usePredictImage();
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <Camera className="w-5 h-5" />
            <span>Análisis en Tiempo Real</span>
          </CardTitle>
          <Button
            onClick={() => setIsAnalyzing(!isAnalyzing)}
            variant={isAnalyzing ? "destructive" : "default"}
            size="sm"
          >
            {isAnalyzing ? (
              <>
                <Pause className="w-4 h-4 mr-2" />
                Pausar
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Iniciar
              </>
            )}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Cámara o Archivo */}
        <CameraOrFilePicker
          onPick={(file) => {
            mutate(file); // envía al backend
          }}
        />

        {/* Estado de petición */}
        {isPending && <Badge className="bg-blue-600 text-white">Analizando imagen...</Badge>}
        {error && <p className="text-sm text-red-600">{error.message}</p>}

        {/* Resultado del backend (usa tu UI) */}
        {data && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Etiqueta:</span>
              <Badge>{data.label}</Badge>
            </div>

            <div>
              <h4 className="font-medium mb-2">Scores</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {Object.entries(data.scores).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between rounded border p-2">
                    <span className="text-sm">{k}</span>
                    <Badge variant="outline">{v.toFixed(4)}</Badge>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="font-medium mb-2">
                Recomendaciones ({data.recomendaciones.label_principal})
              </h4>
              <div className="space-y-3">
                {Object.entries(data.recomendaciones.detalles).map(([k, v]) => (
                  <div key={k} className="rounded border p-3">
                    <div className="flex items-center justify-between mb-2">
                      <strong>{k}</strong>
                      <Badge variant="outline">
                        {v.severity} · {v.score.toFixed(4)}
                      </Badge>
                    </div>
                    <ul className="list-disc pl-5 text-sm space-y-1">
                      {v.acciones.map((a, i) => (
                        <li key={i}>{a}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ... puedes mantener tus “Estados Cognitivos Detectados”/recomendaciones mock si quieres */}
      </CardContent>
    </Card>
  );
}
