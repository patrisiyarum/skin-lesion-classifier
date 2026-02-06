export default function GradCam({ gradCam }) {
  const panels = [
    { src: gradCam.original, label: "Preprocessed Input" },
    { src: gradCam.heatmap, label: "Grad-CAM Heatmap" },
    { src: gradCam.overlay, label: "Overlay" },
  ];

  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-800 mb-3 pb-2 border-b-2 border-slate-100">
        Grad-CAM Visual Explanation
      </h2>

      <div className="p-3 rounded-r-lg text-sm text-slate-700 bg-slate-50 border-l-4 border-blue-500 mb-4">
        <strong>What is Grad-CAM?</strong> &mdash; Gradient-weighted Class
        Activation Mapping highlights the regions of the image that most
        influenced the model&apos;s prediction. Warm colours (red/yellow) indicate
        high-importance areas.
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {panels.map(({ src, label }) => (
          <div key={label} className="text-center">
            <img
              src={src}
              alt={label}
              className="w-full rounded-lg shadow-sm border border-slate-100"
            />
            <p className="text-xs text-slate-500 mt-2">{label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
