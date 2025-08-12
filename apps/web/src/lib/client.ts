export const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export const client = (path: string) => {
  const base = API_URL.endsWith("/") ? API_URL : API_URL + "/";
  const p = path.startsWith("/") ? path.slice(1) : path;
  return new URL(p, base).toString();
};
