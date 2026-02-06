import { useCallback, useRef, useState } from "react";

export default function ImageUpload({ onFileSelected, disabled }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFile = useCallback(
    (file) => {
      if (file && file.type.startsWith("image/")) {
        onFileSelected(file);
      }
    },
    [onFileSelected]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragActive(false);
      if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
    },
    [handleFile]
  );

  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-800 mb-3 pb-2 border-b-2 border-slate-100">
        Upload Image
      </h2>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          border-2 border-dashed rounded-xl p-10 text-center cursor-pointer
          transition-colors duration-200
          ${
            dragActive
              ? "border-blue-500 bg-blue-50"
              : "border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50/50"
          }
          ${disabled ? "opacity-50 pointer-events-none" : ""}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.[0]) handleFile(e.target.files[0]);
          }}
        />
        <div className="text-4xl mb-3">📷</div>
        <p className="text-slate-600 font-medium">
          Drag & drop an image here, or click to browse
        </p>
        <p className="text-xs text-slate-400 mt-2">
          JPG, JPEG, PNG, BMP supported &middot; Resized to 224&times;224 for
          inference
        </p>
      </div>
    </div>
  );
}
