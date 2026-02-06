export const CLASS_INFO = {
  benign: {
    fullName: "Benign Lesion",
    description:
      "The lesion is likely non-cancerous. Common benign lesions include " +
      "melanocytic nevi (moles), seborrheic keratoses, dermatofibromas, " +
      "and vascular lesions. Regular monitoring is still recommended.",
    risk: "Low risk",
    color: "#27ae60",
  },
  malignant: {
    fullName: "Malignant Lesion",
    description:
      "The lesion shows characteristics associated with skin cancer, " +
      "including melanoma, basal cell carcinoma, or actinic keratoses. " +
      "Prompt consultation with a dermatologist is strongly recommended.",
    risk: "High risk",
    color: "#e74c3c",
  },
};

export const CLASS_NAMES = ["benign", "malignant"];
