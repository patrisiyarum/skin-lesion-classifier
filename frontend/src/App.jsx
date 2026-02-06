import { useState, useCallback } from "react";
import Hero from "./components/Hero";
import ImageUpload from "./components/ImageUpload";
import PredictionResult from "./components/PredictionResult";
import Probabilities from "./components/Probabilities";
import GradCam from "./components/GradCam";
import ClassInfo from "./components/ClassInfo";

export default function App() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFile = useCallback(async (file) => {
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch("/api/predict", { method: "POST", body: form });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Server error (${res.status})`);
      }

      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="flex">
        {/* ---- Sidebar ---- */}
        <aside className="hidden lg:block w-72 shrink-0 border-r border-slate-200 bg-white px-5 py-8 sticky top-0 h-screen overflow-y-auto">
          <h2 className="text-xl font-bold text-slate-800 mb-6">
            Skin&nbsp;Lesion Classifier
          </h2>
          <ClassInfo />
          <p className="text-[11px] text-slate-400 mt-8 leading-relaxed">
            For educational and portfolio demonstration purposes only. Not a
            medical diagnostic device.
          </p>
        </aside>

        {/* ---- Main content ---- */}
        <main className="flex-1 px-4 sm:px-8 py-8 max-w-5xl mx-auto w-full">
          <Hero />

          {/* Upload + preview row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <ImageUpload onFileSelected={handleFile} disabled={loading} />

            {preview && (
              <div>
                <h2 className="text-lg font-semibold text-slate-800 mb-3 pb-2 border-b-2 border-slate-100">
                  Uploaded Image
                </h2>
                <img
                  src={preview}
                  alt="Uploaded preview"
                  className="w-full rounded-xl shadow-sm border border-slate-100 object-cover"
                />
              </div>
            )}
          </div>

          {/* Loading state */}
          {loading && (
            <div className="text-center py-12">
              <div className="inline-block h-10 w-10 border-4 border-slate-200 border-t-blue-500 rounded-full animate-spin" />
              <p className="mt-3 text-slate-500 text-sm">
                Running inference&hellip;
              </p>
            </div>
          )}

          {/* Error state */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 mb-8 text-sm">
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Results */}
          {result && (
            <>
              <hr className="border-slate-200 mb-8" />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <PredictionResult result={result} />
                <Probabilities
                  probabilities={result.probabilities}
                  predicted={result.class_name}
                />
              </div>

              {result.grad_cam && (
                <>
                  <hr className="border-slate-200 mb-8" />
                  <GradCam gradCam={result.grad_cam} />
                </>
              )}

              <p className="text-center text-xs text-slate-400 mt-12 pb-4">
                This tool is for{" "}
                <strong>educational and portfolio demonstration purposes only</strong>{" "}
                and is not a medical diagnostic device. Always consult a
                qualified dermatologist.
              </p>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
