import { useRef, useState } from "react";
import { useUserMedia } from "@/hooks/useCamara";
import { captureVideoFrameAsFile } from "@/components/face/capturar";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Camera, ImageUp, Pause, Play, RefreshCw } from "lucide-react";

type Props = {
  onPick: (file: File) => void;
};

export function CameraOrFilePicker({ onPick }: Props) {
  const [tab, setTab] = useState<"camera" | "file">("camera");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const { videoRef, ready, error, start, stop } = useUserMedia({
    video: { facingMode: { ideal: "environment" } }, 
    audio: false,
  });

  return (
    <Card>
      <CardContent className="p-4 space-y-4">
        <div className="flex gap-2">
          <Button variant={tab === "camera" ? "default" : "outline"} onClick={() => setTab("camera")} size="sm">
            <Camera className="w-4 h-4 mr-2" /> Cámara
          </Button>
          <Button variant={tab === "file" ? "default" : "outline"} onClick={() => setTab("file")} size="sm">
            <ImageUp className="w-4 h-4 mr-2" /> Archivo
          </Button>
        </div>

        {tab === "camera" ? (
          <div className="space-y-3">
            <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="w-full h-full object-contain bg-black"
              />
              {ready && (
                <div className="absolute top-2 left-2">
                  <Badge className="bg-red-600 text-white">● LIVE</Badge>
                </div>
              )}
            </div>

            {error && <p className="text-sm text-red-600">{error}</p>}

            <div className="flex items-center gap-2">
              {!ready ? (
                <Button onClick={start} type="button">
                  <Play className="w-4 h-4 mr-2" /> Iniciar cámara
                </Button>
              ) : (
                <Button onClick={stop} variant="destructive" type="button">
                  <Pause className="w-4 h-4 mr-2" /> Detener
                </Button>
              )}

              <Button
                type="button"
                onClick={async () => {
                  if (!videoRef.current) return;
                  const file = await captureVideoFrameAsFile(videoRef.current, "captura.jpg");
                  onPick(file);
                }}
                disabled={!ready}
              >
                <RefreshCw className="w-4 h-4 mr-2" /> Capturar & Enviar
              </Button>
            </div>

          </div>
        ) : (
          <div className="space-y-3">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              // En móviles abre cámara directamente
              capture="environment"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onPick(f);
              }}
            />
            <p className="text-xs text-muted-foreground">
              También puedes tomar foto desde el móvil con el selector (usa <code>capture</code>).
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
