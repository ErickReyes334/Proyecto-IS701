export async function captureVideoFrameAsFile(video: HTMLVideoElement, filename = "capture.jpg", quality = 0.92) {
  const canvas = document.createElement("canvas");
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) throw new Error("Video no listo aún");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("No se pudo crear contexto canvas");
  ctx.drawImage(video, 0, 0, w, h);

  const blob: Blob = await new Promise((res) => canvas.toBlob((b) => res(b as Blob), "image/jpeg", quality));
  return new File([blob], filename, { type: "image/jpeg" });
}
