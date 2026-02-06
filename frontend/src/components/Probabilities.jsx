import { CLASS_INFO } from "../data/classInfo";

export default function Probabilities({ probabilities, predicted }) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-800 mb-3 pb-2 border-b-2 border-slate-100">
        Class Probabilities
      </h2>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-100 space-y-3">
        {probabilities.map(({ name, prob }) => {
          const isPredicted = name === predicted;
          const info = CLASS_INFO[name];
          const barColor = info?.color ?? "#4a90d9";
          const pct = (prob * 100).toFixed(1);

          return (
            <div
              key={name}
              className={`flex items-center gap-3 transition-opacity ${
                isPredicted ? "opacity-100" : "opacity-50"
              }`}
            >
              <span className="w-24 text-sm font-semibold text-slate-700 capitalize">
                {name}
              </span>
              <div className="flex-1 bg-slate-100 rounded-full h-2.5 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${pct}%`, backgroundColor: barColor }}
                />
              </div>
              <span className="w-14 text-right text-sm text-slate-500 tabular-nums">
                {pct}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
