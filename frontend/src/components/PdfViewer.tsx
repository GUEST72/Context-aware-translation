import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

export default function PdfViewer({
  fileUrl,
  pageNumber,
  width,
  onLoadSuccess,
  onLoadError,
}: {
  fileUrl: string;
  pageNumber: number;
  width: number;
  onLoadSuccess: (numPages: number) => void;
  onLoadError: (message: string) => void;
}) {
  return (
    <Document
      file={fileUrl}
      loading={
        <div className="flex h-[70vh] w-[520px] max-w-full items-center justify-center text-sm text-muted-foreground">
          <span className="skeleton-shimmer h-6 w-40 rounded-full" />
        </div>
      }
      onLoadSuccess={(pdf) => onLoadSuccess(pdf.numPages)}
      onLoadError={(error) => onLoadError(String(error))}
    >
      <Page
        pageNumber={pageNumber}
        width={width}
        renderTextLayer
        renderAnnotationLayer={false}
      />
    </Document>
  );
}
