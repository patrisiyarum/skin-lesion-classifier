import { CLASS_INFO, CLASS_NAMES } from "../data/classInfo";

export default function ClassInfo() {
  return (
    <div className="space-y-6">
      {/* Model info */}
      <div>
        <h3 className="font-semibold text-slate-700 mb-2">About the Model</h3>
        <dl className="text-sm text-slate-600 space-y-1">
          <div>
            <dt className="inline font-medium">Architecture:</dt>{" "}
            <dd className="inline">EfficientNet-B0</dd>
          </div>
          <div>
            <dt className="inline font-medium">Training data:</dt>{" "}
            <dd className="inline">HAM10000</dd>
          </div>
          <div>
            <dt className="inline font-medium">Task:</dt>{" "}
            <dd className="inline">Binary (benign vs malignant)</dd>
          </div>
          <div>
            <dt className="inline font-medium">Input:</dt>{" "}
            <dd className="inline">224 &times; 224 dermoscopic image</dd>
          </div>
        </dl>
      </div>

      <hr className="border-slate-200" />

      {/* Classes */}
      <div>
        <h3 className="font-semibold text-slate-700 mb-3">
          Classification Categories
        </h3>
        <div className="space-y-3">
          {CLASS_NAMES.map((name) => {
            const info = CLASS_INFO[name];
            return (
              <div key={name}>
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="font-semibold text-sm text-slate-800">
                    {info.fullName}
                  </span>
                  <span
                    className="text-[11px] px-2 py-0.5 rounded-full text-white font-medium"
                    style={{ backgroundColor: info.color }}
                  >
                    {info.risk}
                  </span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  {info.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
