import { client } from "./client";
import { type PredictResponse } from "./types";
import { useMutation } from "@tanstack/react-query";

export async function predictImage(file: File): Promise<PredictResponse> {
  const form = new FormData();
  form.append("imagen", file);

  const res = await fetch(client("/api/predict-image/"), {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Error ${res.status}: ${await res.text()}`);
  return res.json();
}

export function usePredictImage() {
  return useMutation<PredictResponse, Error, File>({
    mutationFn: (file: any) => predictImage(file),
  });
}
