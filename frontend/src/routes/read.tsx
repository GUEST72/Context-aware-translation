import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  BookOpen,
  ChevronLeft,
  ChevronRight,
  FileText,
  Languages,
  Plus,
  Sparkles,
  Star,
  Upload,
} from "lucide-react";
import { Button } from "@/components/button";
import {
  makeLocalId,
  makeLocalTimestamp,
  useLibrary,
  type TranslationHistoryItem,
} from "@/lib/library-store";
import { requestTranslation } from "@/lib/translationApi";
import { cn } from "@/lib/utils";

const PdfViewer = lazy(() => import("@/components/PdfViewer"));

export const Route = createFileRoute("/read")({
  head: () => ({
    meta: [
      { title: "Reading Room — LexisFlow" },
      { name: "description", content: "Read and translate your academic book in a focused reading room." },
    ],
  }),
  component: Reader,
});

interface SelectionState {
  text: string;
  pageNumber: number;
  x: number;
  y: number;
}

function Reader() {
  const navigate = useNavigate();
  const { pdfUrl, fileName, history, addTranslation, toggleStar } = useLibrary();

  const [mounted, setMounted] = useState(false);
  const [numPages, setNumPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageWidth, setPageWidth] = useState(680);
  const [selection, setSelection] = useState<SelectionState | null>(null);
  const [latest, setLatest] = useState<TranslationHistoryItem | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);
  const [apiError, setApiError] = useState("");

  const viewerRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    const update = () => {
      if (!viewerRef.current) return;
      setPageWidth(Math.max(320, Math.min(820, Math.floor(viewerRef.current.clientWidth * 0.82))));
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [mounted, pdfUrl]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const canTranslate = useMemo(() => !!selection?.text && !isTranslating, [selection, isTranslating]);

  function handleSelection() {
    const sel = window.getSelection();
    const text = sel?.toString().trim() ?? "";
    if (!text || !sel || sel.rangeCount === 0) {
      setSelection(null);
      return;
    }
    const range = sel.getRangeAt(0);
    if (!viewerRef.current || !viewerRef.current.contains(range.commonAncestorContainer)) return;
    const rect = range.getBoundingClientRect();
    const containerRect = viewerRef.current.getBoundingClientRect();
    setSelection({
      text,
      pageNumber: currentPage,
      x: rect.left - containerRect.left + rect.width / 2,
      y: rect.top - containerRect.top - 14,
    });
  }

  async function handleTranslate() {
    if (!selection?.text) return;
    setIsTranslating(true);
    setApiError("");
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const data = await requestTranslation(
        { text: selection.text, page_number: selection.pageNumber },
        controller.signal,
      );
      if (data.error) return setApiError(data.error);
      if (!data.translation) return setApiError("The server did not return a translation.");
      const entry: TranslationHistoryItem = {
        id: makeLocalId(),
        original: selection.text,
        translated: data.translation,
        pageNumber: selection.pageNumber,
        timestamp: makeLocalTimestamp(),
      };
      setLatest(entry);
      addTranslation(entry);
      setSelection(null);
      window.getSelection()?.removeAllRanges();
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setApiError("Translation request failed. Please try again.");
    } finally {
      setIsTranslating(false);
    }
  }

  if (!pdfUrl) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-6 parchment px-6 text-center">
        <span className="flex h-16 w-16 items-center justify-center rounded-2xl [background:var(--gradient-navy)] shadow-elegant">
          <BookOpen className="h-7 w-7 text-primary-foreground" />
        </span>
        <div>
          <h1 className="font-serif text-3xl font-semibold text-foreground">Your reading room is empty</h1>
          <p className="mt-2 max-w-sm text-muted-foreground">
            Upload an academic PDF to begin reading and translating.
          </p>
        </div>
        <Button variant="gold" size="lg" asChild>
          <Link to="/">
            <Upload className="h-4 w-4" /> Upload a book
          </Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Reader top bar */}
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-border bg-card px-4 md:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <Link to="/" className="flex shrink-0 items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg [background:var(--gradient-navy)]">
              <BookOpen className="h-4 w-4 text-primary-foreground" />
            </span>
            <span className="hidden font-serif text-lg font-semibold text-foreground sm:block">LexisFlow</span>
          </Link>
          <span className="hidden h-6 w-px bg-border md:block" />
          <div className="flex min-w-0 items-center gap-2 rounded-full border border-border bg-secondary/60 px-3 py-1.5">
            <FileText className="h-3.5 w-3.5 shrink-0 text-accent-foreground" />
            <span className="truncate text-xs font-medium text-foreground">{fileName}</span>
          </div>
        </div>
        <Button variant="outline" size="sm" asChild>
          <Link to="/">
            <Plus className="h-4 w-4" /> New book
          </Link>
        </Button>
      </header>

      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* PDF viewer */}
        <section className="relative flex min-h-0 flex-1 flex-col parchment">
          <div
            ref={viewerRef}
            onMouseUp={handleSelection}
            className="relative flex flex-1 items-start justify-center overflow-auto p-4 md:p-10"
          >
            <div className="relative rounded-xl border border-border bg-cream p-3 shadow-book md:p-5">
              {mounted ? (
                <Suspense
                  fallback={
                    <div className="flex h-[70vh] w-[520px] max-w-full items-center justify-center">
                      <span className="skeleton-shimmer h-6 w-40 rounded-full" />
                    </div>
                  }
                >
                  <PdfViewer
                    fileUrl={pdfUrl}
                    pageNumber={currentPage}
                    width={pageWidth}
                    onLoadSuccess={(n) => {
                      setNumPages(n);
                      setCurrentPage((p) => Math.min(Math.max(p, 1), n));
                    }}
                    onLoadError={(msg) => setApiError(`Unable to render PDF: ${msg}`)}
                  />
                </Suspense>
              ) : (
                <div className="flex h-[70vh] w-[520px] max-w-full items-center justify-center">
                  <span className="skeleton-shimmer h-6 w-40 rounded-full" />
                </div>
              )}

              <AnimatePresence>
                {selection && (
                  <motion.button
                    type="button"
                    initial={{ opacity: 0, scale: 0.85, y: 6 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.85 }}
                    transition={{ type: "spring", stiffness: 400, damping: 26 }}
                    onClick={handleTranslate}
                    disabled={!canTranslate}
                    className="absolute z-30 inline-flex -translate-x-1/2 -translate-y-full items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant [background:var(--gradient-navy)] disabled:opacity-60"
                    style={{ left: selection.x, top: selection.y }}
                  >
                    <Languages className="h-4 w-4 text-gold" />
                    {isTranslating ? "Translating…" : "Translate"}
                  </motion.button>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Page controls */}
          <div className="pointer-events-none absolute bottom-6 left-1/2 z-20 -translate-x-1/2">
            <div className="pointer-events-auto flex items-center gap-6 rounded-full glass-panel px-6 py-3 shadow-elegant">
              <button
                type="button"
                onClick={() => {
                  setCurrentPage((p) => Math.max(1, p - 1));
                  setSelection(null);
                }}
                disabled={currentPage <= 1}
                className="group flex items-center gap-1 text-muted-foreground transition-colors hover:text-foreground disabled:opacity-40"
              >
                <ChevronLeft className="h-5 w-5 transition-transform group-hover:-translate-x-0.5" />
              </button>
              <span className="text-xs font-semibold text-foreground">
                Page {currentPage} <span className="text-muted-foreground">/ {numPages || "…"}</span>
              </span>
              <button
                type="button"
                onClick={() => {
                  setCurrentPage((p) => Math.min(numPages || 1, p + 1));
                  setSelection(null);
                }}
                disabled={numPages === 0 || currentPage >= numPages}
                className="group flex items-center gap-1 text-muted-foreground transition-colors hover:text-foreground disabled:opacity-40"
              >
                <ChevronRight className="h-5 w-5 transition-transform group-hover:translate-x-0.5" />
              </button>
            </div>
          </div>
        </section>

        {/* Translation panel */}
        <aside className="flex w-full shrink-0 flex-col border-t border-border bg-card lg:h-full lg:w-[380px] lg:border-l lg:border-t-0">
          <div className="flex items-center justify-between border-b border-border px-6 py-5">
            <div className="flex items-center gap-2.5">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-secondary text-primary">
                <Languages className="h-4.5 w-4.5" />
              </span>
              <div>
                <h2 className="font-serif text-base font-semibold text-foreground">Translation</h2>
                <span className="flex items-center gap-1 text-[11px] text-accent-foreground">
                  <Sparkles className="h-3 w-3" /> Context-aware engine
                </span>
              </div>
            </div>
            <span
              className={cn(
                "h-2 w-2 rounded-full",
                isTranslating ? "animate-pulse bg-accent" : "bg-success",
              )}
            />
          </div>

          {apiError && (
            <div className="mx-6 mt-4 rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-2.5 text-xs text-destructive">
              {apiError}
            </div>
          )}

          <div className="flex-1 space-y-8 overflow-y-auto px-6 py-6">
            {/* Latest */}
            <section>
              <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                Latest passage
              </h3>
              {isTranslating ? (
                <div className="space-y-3 rounded-2xl border border-border bg-secondary/40 p-5">
                  <div className="skeleton-shimmer h-3 w-3/4 rounded-full" />
                  <div className="skeleton-shimmer h-3 w-full rounded-full" />
                  <div className="skeleton-shimmer h-3 w-5/6 rounded-full" />
                </div>
              ) : latest ? (
                <TranslationCard item={latest} onStar={() => toggleStar(latest.id)} />
              ) : (
                <div className="rounded-2xl border border-dashed border-border bg-secondary/30 p-6 text-center">
                  <Languages className="mx-auto mb-3 h-6 w-6 text-muted-foreground" />
                  <p className="text-sm font-medium text-foreground">Highlight to translate</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Select any text on the page, then tap Translate.
                  </p>
                </div>
              )}
            </section>

            {/* History */}
            {history.length > 0 && (
              <section>
                <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                  History
                </h3>
                <div className="space-y-3">
                  {history.map((item) => (
                    <TranslationCard key={item.id} item={item} compact onStar={() => toggleStar(item.id)} />
                  ))}
                </div>
              </section>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

function TranslationCard({
  item,
  compact = false,
  onStar,
}: {
  item: TranslationHistoryItem;
  compact?: boolean;
  onStar: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "rounded-2xl border border-border bg-background p-5",
        !compact && "shadow-soft",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <p className="line-clamp-2 flex-1 text-xs italic text-muted-foreground">"{item.original}"</p>
        <button
          type="button"
          onClick={onStar}
          className={cn(
            "shrink-0 rounded-full p-1 transition-colors",
            item.starred ? "text-accent" : "text-muted-foreground hover:text-accent",
          )}
          aria-label="Star translation"
        >
          <Star className={cn("h-3.5 w-3.5", item.starred && "fill-current")} />
        </button>
      </div>
      <p className={cn("mt-3 font-medium leading-relaxed text-foreground", compact ? "text-sm" : "text-[15px]")}>
        {item.translated}
      </p>
      <div className="mt-3 flex items-center justify-between text-[11px] text-muted-foreground">
        <span>Page {item.pageNumber}</span>
        <span>{item.timestamp}</span>
      </div>
    </motion.div>
  );
}
