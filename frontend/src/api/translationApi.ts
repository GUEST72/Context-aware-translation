export interface TranslateRequest {
  text: string
  page_number: number
}

export interface TranslateResponse {
  translation?: string
  error?: string
}

export interface UploadPdfResponse {
  message: string
  filename: string
  file_path: string
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

/**
 * IMPORTANT NOTE:
 * A future backend endpoint will parse uploaded PDFs and return structured JSON.
 * That endpoint is not implemented yet.
 * Current integration must continue sending highlighted text + page_number only.
 * Keep this module as the integration boundary so future JSON parsing can be
 * introduced with minimal UI/component changes.
 */
export async function requestTranslation(
  payload: TranslateRequest,
  signal?: AbortSignal,
): Promise<TranslateResponse> {
  const response = await fetch(`${API_BASE_URL}/Translate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    signal,
  })

  if (!response.ok) {
    throw new Error(`Translation request failed with status ${response.status}`)
  }

  return (await response.json()) as TranslateResponse
}

export async function uploadPdf(file: File): Promise<UploadPdfResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/upload_pdf`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || `Upload failed with status ${response.status}`)
  }

  return (await response.json()) as UploadPdfResponse
}
