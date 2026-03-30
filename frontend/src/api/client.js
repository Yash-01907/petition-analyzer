// api/client.js — axios API calls to the FastAPI backend
// The Vite proxy at /api forwards to http://localhost:8000

import axios from "axios";

const api = axios.create({
  baseURL: "/",
  headers: { "Content-Type": "application/json" },
});

export async function analyzeCSV(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await api.post("/api/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function retrainCSV(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await api.post("/api/retrain", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function scoreDraft(draft) {
  const res = await api.post("/api/score-draft", draft);
  return res.data;
}

export async function getSampleCSV({ bustCache = false } = {}) {
  const res = await api.get("/api/sample-csv", {
    responseType: "blob",
    params: bustCache ? { _t: Date.now() } : undefined,
    headers: bustCache
      ? {
          "Cache-Control": "no-cache, no-store, must-revalidate",
          Pragma: "no-cache",
          Expires: "0",
        }
      : undefined,
  });
  return res.data;
}

export async function retrainAndRefresh(file) {
  const retrainResult = await retrainCSV(file);
  const analysis = retrainResult?.analysis;
  if (!analysis) {
    throw new Error("Retrain completed but no analysis payload was returned.");
  }
  return { analysis, retrainResult };
}

export default api;
