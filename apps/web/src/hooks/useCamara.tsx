
import { useEffect, useRef, useState } from "react";

type Constraints = MediaStreamConstraints;

export function useUserMedia(constraints: Constraints = { video: { facingMode: "user" }, audio: false }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = async () => {
    try {
      setError(null);
      streamRef.current = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = streamRef.current;
        await videoRef.current.play().catch(() => {});
      }
      setReady(true);
    } catch (e: any) {
      setError(e?.message ?? "No se pudo iniciar la cámara");
      setReady(false);
    }
  };

  const stop = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setReady(false);
  };

  useEffect(() => {
    return () => stop(); // cleanup al desmontar
  }, []);

  return { videoRef, ready, error, start, stop };
}
