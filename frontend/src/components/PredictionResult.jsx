export default function PredictionResult({ result }) {
  const { full_name, risk, risk_color, confidence, description } = result;
  const pct = (confidence * 100).toFixed(1);

  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-800 mb-3 pb-2 border-b-2 border-slate-100">
        Prediction Result
      </h2>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-100">
        {/* Prediction badge */}
        <span
          className="inline-block px-4 py-2 rounded-lg font-bold text-lg mb-2"
          style={{ backgroundColor: risk_color + "18", color: risk_color }}
        >
          {full_name}
        </span>

        <br />

        {/* Risk badge */}
        <span
          className="inline-block px-3 py-1 rounded-full text-xs font-semibold text-white"
          style={{ backgroundColor: risk_color }}
        >
          {risk}
        </span>

        {/* Confidence */}
        <div className="mt-5">
          <span className="text-sm text-slate-500">Confidence</span>
          <div className="text-3xl font-bold text-slate-900">{pct}%</div>
          <div className="w-full bg-slate-200 rounded-full h-3 mt-1 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700 ease-out"
              style={{ width: `${pct}%`, backgroundColor: risk_color }}
            />
          </div>
        </div>

        {/* Description */}
        <div
          className="mt-4 p-3 rounded-r-lg text-sm text-slate-700 bg-slate-50 border-l-4"
          style={{ borderLeftColor: risk_color }}
        >
          {description}
        </div>
      </div>
    </div>
  );
}
