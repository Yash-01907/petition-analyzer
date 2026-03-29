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

export async function scoreDraft(draft) {
  const res = await api.post("/api/score-draft", draft);
  return res.data;
}

export async function getSampleData() {
  const res = await api.get("/api/sample-data");
  return res.data;
}

export default api;
