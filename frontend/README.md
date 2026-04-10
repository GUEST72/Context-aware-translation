# Context-Aware Translation Frontend

This frontend integrates the new UI design with the existing backend translation API.

## Current Workflow

1. User uploads a PDF in the browser.
2. User views pages in the PDF workspace.
3. User highlights text on a page.
4. Frontend sends `text` and `page_number` to backend `POST /Translate`.
5. Frontend displays returned translation and saves it in translation history.

## API Contract (Current)

Request body:

```json
{
	"text": "selected text",
	"page_number": 1
}
```

Response body:

```json
{
	"translation": "..."
}
```

or

```json
{
	"error": "Text not found"
}
```

## Important Future Feature (Not Implemented Yet)

IMPORTANT NOTE:

There is a backend API that does not currently exist but will be implemented later.

That future API will parse uploaded PDFs and return structured JSON with extracted text and metadata (for example page numbers and positions).

For now:

- Keep using the current workflow (`text` + `page_number`).
- Do not use IDs in request payloads.
- Keep frontend code structured so future parsed-PDF JSON integration can be added with minimal UI changes.

Integration boundary for this is defined in `src/api/translationApi.ts`.

## Local Development

From `frontend/`:

```bash
npm install
npm run dev
```

The dev server proxies `/api/*` to `http://127.0.0.1:8000` via Vite config.

## Quality Checks

```bash
npm run typecheck
npm run build
```
