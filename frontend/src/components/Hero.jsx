export default function Hero() {
  return (
    <div className="rounded-2xl bg-gradient-to-br from-[#0f2027] via-[#203a43] to-[#2c5364] px-6 py-12 text-center text-white mb-8">
      <h1 className="text-3xl sm:text-4xl font-bold mb-2 tracking-tight">
        Skin Lesion Risk Classifier
      </h1>
      <p className="text-sm sm:text-base text-slate-300 max-w-2xl mx-auto leading-relaxed">
        Upload a dermoscopic image to classify the skin lesion as{" "}
        <strong className="text-white">benign or malignant</strong> using a
        deep-learning model with Grad-CAM visual explanations.
      </p>
    </div>
  );
}
