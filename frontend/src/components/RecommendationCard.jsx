// frontend/src/components/RecommendationCard.jsx

const IMPACT_COLORS = {
  high: { bg: "bg-red-50", border: "border-red-200", badge: "bg-red-100 text-red-700" },
  medium: { bg: "bg-yellow-50", border: "border-yellow-200", badge: "bg-yellow-100 text-yellow-700" },
  low: { bg: "bg-blue-50", border: "border-blue-200", badge: "bg-blue-100 text-blue-700" },
};

export default function RecommendationCard({ rec, rank }) {
  const colors = IMPACT_COLORS[rec.grade_impact] || IMPACT_COLORS.low;

  return (
    <div className={`rounded-xl border ${colors.bg} ${colors.border} p-5`}>
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-gray-400">#{rank}</span>
          <h4 className="font-semibold text-gray-900 text-sm">{rec.title}</h4>
        </div>
        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${colors.badge} whitespace-nowrap`}>
          {rec.grade_impact.toUpperCase()} IMPACT
        </span>
      </div>

      <p className="text-sm text-gray-700 mb-3">{rec.description}</p>

      {rec.example && (
        <div className="bg-white rounded-lg border border-gray-200 p-3">
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">
            Example
          </div>
          <pre className="text-xs text-gray-700 whitespace-pre-wrap font-sans">
            {rec.example}
          </pre>
        </div>
      )}
    </div>
  );
}
