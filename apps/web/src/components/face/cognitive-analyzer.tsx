import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Camera, Play, Pause } from "lucide-react";
import { CameraOrFilePicker } from "@/components/face/camara";
import { usePredictImage } from "@/lib/check";

type Detalle = {
  severity: "alto" | "medio" | "bajo" | string;
  score: number;
  acciones: string[];
};

type ResponseData = {
  label: string;
  scores: Record<string, number>;
  recomendaciones: {
    label_principal: string;
    detalles: Record<string, Detalle>;
  };
};

export function CognitiveAnalyzer() {
  const { mutate, data, isPending, error } = usePredictImage() as {
    mutate: (file: File) => void;
    data?: ResponseData;
    isPending: boolean;
    error?: { message?: string };
  };

  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  const detalles: Record<string, Detalle> =
    (data?.recomendaciones?.detalles as Record<string, Detalle>) ?? {};
  const entries = Object.entries(detalles);

  const severityRank = (s?: string) =>
    s === "alto" ? 2 : s === "medio" ? 1 : 0;

  const [winnerKey, winnerVal] =
    entries
      .sort((a, b) => {
        const d = (b[1]?.score ?? 0) - (a[1]?.score ?? 0);
        if (d !== 0) return d;
        return severityRank(b[1]?.severity) - severityRank(a[1]?.severity);
      })[0] ?? [];

  const sortedScores = Object.entries(data?.scores ?? {}).sort((a, b) => {
    const bv = Number(b[1] ?? 0);
    const av = Number(a[1] ?? 0);
    if (bv !== av) return bv - av; 
    return a[0].localeCompare(b[0]);
  });

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
            mutate(file);
          }}
        />

        {/* Estado de petición */}
        {isPending && (
          <Badge className="bg-blue-600 text-white">Analizando imagen...</Badge>
        )}
        {error?.message && (
          <p className="text-sm text-red-600">{error.message}</p>
        )}

        {/* Resultado del backend */}
        {data && (
          <div className="space-y-4">
            {/* Etiqueta principal */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Etiqueta:</span>
              <Badge>{data.label}</Badge>
            </div>

            {/* Scores (ordenados desc) */}
            {sortedScores.length > 0 && (
              <div>
                <h4 className="font-medium mb-2">Scores</h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {sortedScores.map(([k, v]) => (
                    <div
                      key={k}
                      className="flex items-center justify-between rounded border p-2"
                    >
                      <span className="text-sm">{k}</span>
                      <Badge variant="outline">{Number(v).toFixed(4)}</Badge>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recomendaciones - SOLO el ganador */}
            <div>
              <h4 className="font-medium mb-2">
                Recomendaciones ({data.recomendaciones.label_principal})
              </h4>

              {winnerKey && winnerVal ? (
                <div className="rounded border p-3">
                  <div className="flex items-center justify-between mb-2">
                    <strong>{winnerKey}</strong>
                    <Badge variant="outline">
                      {winnerVal.severity} · {winnerVal.score.toFixed(4)}
                    </Badge>
                  </div>
                  <ul className="list-disc pl-5 text-sm space-y-1">
                    {winnerVal.acciones.map((a, i) => (
                      <li key={i}>{a}</li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Sin recomendaciones.
                </p>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
