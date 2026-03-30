// frontend/src/App.jsx

import { useState } from "react";
import UploadPanel from "./components/UploadPanel";
import AnalysisDashboard from "./components/AnalysisDashboard";
import DraftScorer from "./components/DraftScorer";

export default function App() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [activeTab, setActiveTab] = useState("upload");
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-4 sm:px-6 py-3 sm:py-4">
        <div className="max-w-7xl mx-auto flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-lg sm:text-2xl font-bold text-gray-900 leading-tight">
              📋 Petition Effectiveness Analyzer
            </h1>
            <p className="text-xs sm:text-sm text-gray-500 mt-1">
              Turn campaign history into actionable copy strategy
            </p>
          </div>
          {analysisResult && (
            <nav className="grid grid-cols-3 gap-2 w-full md:w-auto">
              {["upload", "analysis", "scorer"].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-3 sm:px-4 py-2 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
                    activeTab === tab
                      ? "bg-blue-600 text-white"
                      : "text-gray-600 hover:bg-gray-100"
                  }`}
                >
                  {tab === "upload" ? "📂 Data" :
                   tab === "analysis" ? "📊 Analysis" : "✏️ Score Draft"}
                </button>
              ))}
            </nav>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-8">
        {activeTab === "upload" && (
          <UploadPanel
            onAnalysisComplete={(result) => {
              setAnalysisResult(result);
              setActiveTab("analysis");
            }}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        )}
        {activeTab === "analysis" && analysisResult && (
          <AnalysisDashboard
            result={analysisResult}
            onAnalysisComplete={(updatedResult) => setAnalysisResult(updatedResult)}
          />
        )}
        {activeTab === "scorer" && (
          <DraftScorer avgRate={analysisResult?.summary?.avg_conversion_rate} />
        )}
      </main>
    </div>
  );
}
