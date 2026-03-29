// frontend/src/components/AnalysisDashboard.jsx

import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts";

const GRADE_COLORS = {
  A: "bg-green-100 text-green-800",
  B: "bg-green-50 text-green-700",
  C: "bg-yellow-50 text-yellow-700",
  D: "bg-orange-50 text-orange-700",
  F: "bg-red-50 text-red-800",
};

export default function AnalysisDashboard({ result }) {
  const { summary, feature_importance, campaign_scores, archetypes, source_breakdown } = result;
  const [downloading, setDownloading] = useState(false);

  const handleDownloadPDF = async () => {
    setDownloading(true);
    try {
      const response = await fetch("/api/export-pdf");
      if (!response.ok) throw new Error("PDF generation failed");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "petition_analysis_report.pdf";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("PDF download failed:", err);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header with Download */}
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900">Analysis Results</h2>
        <button
          onClick={handleDownloadPDF}
          disabled={downloading}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
        >
          {downloading ? (
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
          ) : (
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
          )}
          {downloading ? "Generating..." : "Download PDF Report"}
        </button>
      </div>

      {/* Top Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-1">Campaigns Analyzed</h3>
          <p className="text-3xl font-bold text-gray-900">{summary.n_campaigns}</p>
        </div>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-1">Avg Conversion Rate</h3>
          <p className="text-3xl font-bold text-gray-900">{summary.avg_conversion_rate}%</p>
        </div>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-1">Top Performing Traffic</h3>
          <p className="text-3xl font-bold text-gray-900 capitalize">
            {source_breakdown.reduce((max, s) => s.avg_conversion > max.avg_conversion ? s : max, source_breakdown[0])?.traffic_source}
          </p>
        </div>
      </div>

      {/* Feature Importance Chart */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">What Drives Conversation</h3>
        <p className="text-sm text-gray-500 mb-6">
          The machine learning model identified these features as having the highest impact on conversion rates in your historical data.
        </p>
        <div className="h-80 w-full text-sm">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={feature_importance}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 200, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
              <XAxis type="number" hide />
              <YAxis 
                type="category" 
                dataKey="label" 
                tick={{ fontSize: 12, fill: "#4B5563" }} 
                width={190} 
                axisLine={false} 
                tickLine={false} 
              />
              <Tooltip 
                cursor={{ fill: "#F3F4F6" }}
                contentStyle={{ borderRadius: "8px", border: "1px solid #E5E7EB", boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)" }}
                formatter={(value) => [`${value}% importance`, "Impact"]}
              />
              <Bar dataKey="importance_pct" fill="#3B82F6" radius={[0, 4, 4, 0]} barSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Archetypes */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Audience Archetypes</h3>
          <p className="text-sm text-gray-500 mb-4">
            Campaigns clustered by linguistic and structural similarities.
          </p>
          <div className="space-y-4">
            {archetypes.map((arch, index) => (
              <div key={index} className="border border-gray-100 rounded-lg p-4 bg-gray-50">
                <div className="flex justify-between items-center mb-2">
                  <h4 className="font-semibold text-gray-800">{arch.name}</h4>
                  <span className="text-blue-600 font-bold bg-blue-50 px-2 py-1 rounded text-sm">
                    {arch.avg_conversion_rate}% cvr
                  </span>
                </div>
                <div className="text-xs text-gray-500 mb-2">({arch.campaign_count} campaigns)</div>
                <div className="flex flex-wrap gap-1 mb-3">
                  {arch.dominant_traits.map((trait, i) => (
                    <span key={i} className="text-[10px] bg-white border border-gray-200 px-2 py-0.5 rounded-full text-gray-600">
                      {trait}
                    </span>
                  ))}
                </div>
                <div className="text-xs text-gray-600 italic">
                  Example: "{arch.example_headlines[0]}"
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Campaign Scores */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 overflow-hidden flex flex-col">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Campaign Leaderboard</h3>
          <p className="text-sm text-gray-500 mb-4">
            Your recent campaigns graded against the model expectations.
          </p>
          <div className="overflow-y-auto flex-1 max-h-[500px]">
             <table className="min-w-full divide-y divide-gray-200">
               <thead className="bg-gray-50 sticky top-0">
                 <tr>
                   <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Campaign</th>
                   <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Conv %</th>
                   <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Grade</th>
                 </tr>
               </thead>
               <tbody className="bg-white divide-y divide-gray-200">
                 {campaign_scores.slice(0, 10).map((score, index) => (
                   <tr key={index}>
                     <td className="px-3 py-3 whitespace-normal">
                       <span className="text-sm text-gray-900 font-medium line-clamp-2">{score.headline}</span>
                       <span className="text-xs text-gray-400 block">{score.campaign_id} • {score.traffic_source}</span>
                     </td>
                     <td className="px-3 py-3 whitespace-nowrap">
                       <span className="text-sm text-gray-900 font-bold">{score.actual_conversion}%</span>
                     </td>
                     <td className="px-3 py-3 whitespace-nowrap">
                       <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${GRADE_COLORS[score.grade] || "bg-gray-100 text-gray-800"}`}>
                         {score.grade}
                       </span>
                     </td>
                   </tr>
                 ))}
               </tbody>
             </table>
          </div>
        </div>
      </div>
    </div>
  );
}
