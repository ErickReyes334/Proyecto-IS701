import { client } from "./client";
import { type PredictResponse } from "./types";
import { useMutation } from "@tanstack/react-query";

const predictState = async (file: File): Promise<PredictResponse> => {
  const form = new FormData();
  form.append("imagen", file);

  const res = await fetch(`${client}api/predict-image/`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Error ${res.status}: ${text || res.statusText}`);
  }
  return res.json();
};

export function usePredictImage() {
  return useMutation<PredictResponse, Error, File>({
    mutationFn: (file: any) => predictState(file),
  });
}