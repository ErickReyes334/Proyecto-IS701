export interface PredictResponse {
  label: string;
  scores: Record<string, number>;
  emo_features: Record<string, number>;
  recomendaciones: {
    label_principal: string;
    detalles: {
      [k in string]: {
        score: number;
        severity: "bajo" | "medio" | "alto" | string;
        acciones: string[];
        referir_profesional: boolean;
      }
    };
  };
}