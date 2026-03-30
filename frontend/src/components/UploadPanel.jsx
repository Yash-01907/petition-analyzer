// frontend/src/components/UploadPanel.jsx

import React, { useState } from "react";
import { analyzeCSV, getSampleCSV } from "../api/client";

export default function UploadPanel({
  onAnalysisComplete,
  isLoading,
  setIsLoading,
}) {
  const [error, setError] = useState(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await analyzeCSV(file);
      onAnalysisComplete(data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to upload and analyze the CSV file. Please make sure it follows the required format.",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const loadSampleData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const blob = await getSampleCSV();
      const file = new File([blob], "sample_campaigns.csv", {
        type: "text/csv",
      });

      const data = await analyzeCSV(file);
      onAnalysisComplete(data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          err.message ||
          "Failed to load sample data.",
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto mt-6 sm:mt-12 bg-white rounded-xl shadow-sm border border-gray-200 p-5 sm:p-8 text-center">
      <div className="mb-6">
        <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl">
          📊
        </div>
        <h2 className="text-lg sm:text-xl font-semibold text-gray-900">
          Upload Campaign History
        </h2>
        <p className="text-sm text-gray-500 mt-2 max-w-md mx-auto">
          Upload a CSV containing your past petition campaigns to train the
          model. Required columns: headline, body_text, cta_text,
          unique_visitors, signatures, traffic_source.
        </p>
      </div>

      <div className="space-y-4">
        <div>
          <label
            className={`inline-flex w-full sm:w-auto items-center justify-center px-6 py-3 border border-transparent 
                       text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 
                       cursor-pointer transition-colors ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
          >
            <span className="mr-2">📁</span>
            {isLoading ? "Analyzing..." : "Select CSV File"}
            <input
              type="file"
              className="hidden"
              accept=".csv"
              onChange={handleFileUpload}
              disabled={isLoading}
            />
          </label>
        </div>

        <div className="text-sm text-gray-400">or</div>

        <div>
          <button
            onClick={loadSampleData}
            disabled={isLoading}
            className="text-sm font-medium text-blue-600 hover:text-blue-800 disabled:opacity-50"
          >
            Generate & Load Mock Data
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-6 bg-red-50 text-red-700 p-4 rounded-lg text-sm border border-red-200">
          {error}
        </div>
      )}
    </div>
  );
}
