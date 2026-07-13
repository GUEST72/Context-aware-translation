export interface TranslationRequest {
  text: string;
  page_number: number;
}

export interface TranslationResponse {
  translation?: string;
  error?: string;
}

export interface UploadResponse {
  message: string;
  filename: string;
  file_path: string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

export async function requestTranslation(
  payload: TranslationRequest,
  signal?: AbortSignal,
): Promise<TranslationResponse> {
  const response = await fetch(`${API_BASE_URL}/Translate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Translation request failed with status ${response.status}`);
  }

  return (await response.json()) as TranslationResponse;
}

export async function uploadPdf(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload_pdf`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Upload failed with status ${response.status}`);
  }

  return (await response.json()) as UploadResponse;
}
