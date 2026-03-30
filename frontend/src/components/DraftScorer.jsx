// frontend/src/components/DraftScorer.jsx

import { useState } from "react";
import { scoreDraft } from "../api/client";
import RecommendationCard from "./RecommendationCard";

const GRADE_COLORS = {
  A: "bg-green-100 text-green-800 border-green-300",
  B: "bg-green-50 text-green-700 border-green-200",
  C: "bg-yellow-50 text-yellow-700 border-yellow-200",
  D: "bg-orange-50 text-orange-700 border-orange-200",
  F: "bg-red-50 text-red-800 border-red-200",
};

function parseApiError(error) {
  const fallback = "Scoring failed. Upload campaign data first.";
  const detail = error?.response?.data?.detail;

  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail) && detail.length > 0) {
    const messages = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item?.msg && item?.loc) {
          const field = Array.isArray(item.loc)
            ? item.loc[item.loc.length - 1]
            : "field";
          return `${field}: ${item.msg}`;
        }
        if (item?.msg) return item.msg;
        return null;
      })
      .filter(Boolean);

    if (messages.length > 0) {
      return messages.join(" | ");
    }
  }

  if (detail && typeof detail === "object") {
    return detail.message || fallback;
  }

  return error?.message || fallback;
}

export default function DraftScorer({ avgRate }) {
  const [form, setForm] = useState({
    headline: "",
    body_text: "",
    cta_text: "",
    traffic_source: "email",
    cause_category: "environment",
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleScore = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await scoreDraft(form);
      setResult(data);
    } catch (e) {
      setResult(null);
      setError(parseApiError(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-gray-900 mb-1">
          Score Your Draft
        </h2>
        <p className="text-sm text-gray-500 mb-6">
          Paste your campaign copy below to get a predicted conversion rate and
          recommendations before you publish.
        </p>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Headline *
            </label>
            <input
              className="w-full border border-gray-300 rounded-lg px-4 py-2.5 text-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="e.g., Tell Mayor Chen: Stop the Riverside Development Now"
              value={form.headline}
              onChange={(e) => setForm({ ...form, headline: e.target.value })}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Body Text *
            </label>
            <textarea
              className="w-full border border-gray-300 rounded-lg px-4 py-2.5 text-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[140px]"
              placeholder="Paste your full petition body text here..."
              value={form.body_text}
              onChange={(e) => setForm({ ...form, body_text: e.target.value })}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Call-to-Action (CTA) *
            </label>
            <input
              className="w-full border border-gray-300 rounded-lg px-4 py-2.5 text-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="e.g., Tell Mayor Chen: Our River Is Not for Sale"
              value={form.cta_text}
              onChange={(e) => setForm({ ...form, cta_text: e.target.value })}
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Expected Traffic Source
              </label>
              <select
                className="w-full border border-gray-300 rounded-lg px-4 py-2.5 text-sm
                           focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={form.traffic_source}
                onChange={(e) =>
                  setForm({ ...form, traffic_source: e.target.value })
                }
              >
                <option value="email">Email List</option>
                <option value="social">Social Media</option>
                <option value="organic">Organic / SEO</option>
                <option value="paid">Paid Advertising</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Cause Category
              </label>
              <select
                className="w-full border border-gray-300 rounded-lg px-4 py-2.5 text-sm
                           focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={form.cause_category}
                onChange={(e) =>
                  setForm({ ...form, cause_category: e.target.value })
                }
              >
                {[
                  "environment",
                  "housing",
                  "healthcare",
                  "education",
                  "transit",
                  "food_security",
                  "civil_rights",
                  "climate",
                ].map((c) => (
                  <option key={c} value={c}>
                    {c.replace("_", " ").toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
              {error}
            </div>
          )}

          <button
            onClick={handleScore}
            disabled={
              loading || !form.headline || !form.body_text || !form.cta_text
            }
            className="w-full bg-blue-600 text-white rounded-lg py-3 text-sm font-semibold
                       hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed
                       transition-colors"
          >
            {loading ? "Analyzing..." : "Score This Campaign Draft"}
          </button>
        </div>
      </div>

      {result && (
        <div className="space-y-4">
          {/* Quality Warnings */}
          {result.quality_warnings?.length > 0 && (
            <div className="bg-amber-50 border-2 border-amber-300 rounded-xl p-5">
              <h3 className="text-sm font-bold text-amber-800 uppercase tracking-wide mb-3 flex items-center gap-2">
                <span>⚠️</span> Content Quality Issues Detected
              </h3>
              <p className="text-xs text-amber-700 mb-3">
                Your score has been adjusted because the content is outside the
                range of effective petition copy. Fix these issues for an
                accurate prediction.
              </p>
              <div className="space-y-2">
                {result.quality_warnings.map((w, i) => (
                  <div
                    key={i}
                    className={`flex items-start gap-2 text-sm rounded-lg px-3 py-2 ${
                      w.severity === "high"
                        ? "bg-red-100 text-red-800"
                        : "bg-amber-100 text-amber-800"
                    }`}
                  >
                    <span className="font-semibold capitalize shrink-0">
                      {w.field.replace("_", " ")}:
                    </span>
                    <span>{w.issue}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Score Card */}
          <div
            className={`rounded-xl border-2 p-6 ${GRADE_COLORS[result.grade]}`}
          >
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="text-4xl sm:text-5xl font-bold">
                  {result.grade}
                </div>
                <div className="text-lg font-semibold mt-1">{result.label}</div>
                <div className="text-sm mt-2 opacity-80">
                  Predicted conversion:{" "}
                  <strong>{result.predicted_rate}%</strong>
                  {avgRate && ` (your average: ${avgRate}%)`}
                </div>
              </div>
              <div className="text-left sm:text-right text-sm opacity-75">
                <div>
                  Z-score: {result.z_score > 0 ? "+" : ""}
                  {result.z_score}
                </div>
                <div className="text-xs mt-1">
                  vs. your historical campaigns
                </div>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Recommendations ({result.recommendations?.length || 0})
            </h3>
            <div className="space-y-3">
              {result.recommendations?.map((rec, i) => (
                <RecommendationCard key={i} rec={rec} rank={i + 1} />
              ))}
              {(!result.recommendations ||
                result.recommendations.length === 0) && (
                <div
                  className="bg-green-50 border border-green-200 rounded-lg p-4 text-sm
                                text-green-700"
                >
                  ✅ This draft scores well on all major conversion factors.
                  Good work!
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
