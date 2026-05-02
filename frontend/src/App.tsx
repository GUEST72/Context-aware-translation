import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import {
  Archive,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Edit3,
  FileText,
  HelpCircle,
  History,
  Languages,
  Plus,
  Search,
  Settings,
  Share2,
  Star,
  Upload,
} from 'lucide-react'
import { Document, Page, pdfjs } from 'react-pdf'
import { requestTranslation, uploadPdf } from './api/translationApi'
import { cn } from './lib/utils'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

type ViewState = 'upload' | 'workspace'

interface SelectionState {
  text: string
  pageNumber: number
  x: number
  y: number
}

interface TranslationHistoryItem {
  id: string
  original: string
  translated: string
  pageNumber: number
  timestamp: string
  starred?: boolean
}

function makeLocalTimestamp() {
  return new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  })
}

function makeLocalId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }

  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function TopBar({
  view,
  setView,
  fileName,
}: {
  view: ViewState
  setView: (next: ViewState) => void
  fileName: string
}) {
  return (
    <header className="fixed top-0 z-50 flex h-16 w-full items-center justify-between border-b border-outline-variant/10 bg-surface/70 px-8 backdrop-blur-xl">
      <div className="flex items-center gap-8">
        <div className="flex cursor-pointer items-center gap-2" onClick={() => setView('upload')}>
          <div className="primary-gradient flex h-8 w-8 items-center justify-center rounded-lg">
            <BookOpen className="h-5 w-5 text-on-primary" />
          </div>
          <span className="font-headline text-xl font-bold tracking-tighter text-on-surface">LexisFlow</span>
        </div>

        <nav className="hidden items-center gap-6 md:flex">
          {[
            { id: 'documents', label: 'Documents', icon: FileText },
            { id: 'dictionary', label: 'Dictionary', icon: Search },
            { id: 'translate', label: 'Translate', icon: Languages },
          ].map((item) => (
            <button
              key={item.id}
              type="button"
              className={cn(
                'flex items-center gap-2 text-sm font-medium tracking-tight transition-all duration-300',
                view === 'workspace' && item.id === 'documents'
                  ? 'border-b-2 border-primary pb-1 text-primary'
                  : 'text-on-surface-variant hover:text-primary',
              )}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="flex items-center gap-4">
        {fileName ? (
          <div className="hidden max-w-[280px] truncate rounded-full border border-outline-variant/20 bg-surface-container-low px-3 py-1 text-xs text-on-surface-variant md:block">
            {fileName}
          </div>
        ) : null}
        <button
          type="button"
          className="rounded-full p-2 text-on-surface-variant transition-colors hover:bg-surface-container-highest hover:text-primary"
        >
          <Share2 className="h-5 w-5" />
        </button>
        <button
          type="button"
          className="rounded-full p-2 text-on-surface-variant transition-colors hover:bg-surface-container-highest hover:text-primary"
        >
          <Settings className="h-5 w-5" />
        </button>
        <button
          type="button"
          className="rounded-full p-2 text-on-surface-variant transition-colors hover:bg-surface-container-highest hover:text-primary"
        >
          <HelpCircle className="h-5 w-5" />
        </button>
        <div className="ml-2 flex h-8 w-8 items-center justify-center rounded-full border border-outline-variant/20 bg-surface-container-high text-xs font-semibold text-primary">
          LF
        </div>
      </div>
    </header>
  )
}

function UploadScreen({
  onFileSelected,
  uploadError,
  isUploading,
}: {
  onFileSelected: (file: File) => void
  uploadError: string
  isUploading: boolean
}) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  function pickFile() {
    if (!isUploading) {
      inputRef.current?.click()
    }
  }

  function handleFiles(files: FileList | null) {
    const file = files?.[0]
    if (!file) {
      return
    }

    onFileSelected(file)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="relative flex min-h-[calc(100vh-64px)] flex-1 flex-col items-center justify-center px-6"
    >
      <div className="absolute left-1/2 top-1/2 -z-10 h-[600px] w-[600px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary/5 blur-[120px]" />
      <div className="absolute bottom-0 right-0 -z-10 h-[400px] w-[400px] rounded-full bg-secondary/5 blur-[100px]" />

      <div className="w-full max-w-3xl text-center">
        <div className="mb-12">
          <h1 className="mb-6 font-headline text-5xl font-extrabold leading-tight tracking-tighter text-on-surface md:text-7xl">
            Archive your <span className="italic text-primary">knowledge.</span>
          </h1>
          <p className="mx-auto max-w-lg text-lg leading-relaxed text-on-surface-variant">
            Upload a PDF, select text from any page, and translate it instantly with contextual backend processing.
          </p>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(event) => handleFiles(event.target.files)}
        />

        <div
          className="w-full cursor-pointer"
          onClick={pickFile}
          onDragOver={(event) => {
            event.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(event) => {
            event.preventDefault()
            setIsDragging(false)
            handleFiles(event.dataTransfer.files)
          }}
        >
          <div className={cn('rounded-2xl bg-gradient-to-b from-outline-variant/20 to-transparent p-1 transition-all duration-500', isDragging && 'from-primary/40')}>
            <div className="rounded-2xl border border-outline-variant/10 bg-surface-container-low p-12 shadow-2xl transition-all duration-500 hover:bg-surface-container-high/80 md:p-24">
              <div className="mx-auto mb-8 flex h-20 w-20 items-center justify-center rounded-full bg-surface-container-highest ring-1 ring-outline-variant/20 transition-all group-hover:bg-surface-bright">
                <Upload className="h-10 w-10 text-primary" />
              </div>
              <h2 className="mb-3 font-headline text-2xl font-bold text-on-surface">Upload a PDF</h2>
              <p className="mb-8 text-sm uppercase tracking-wide text-on-surface-variant">
                Drag and drop your file here or click below
              </p>
              <button
                type="button"
                disabled={isUploading}
                className={cn(
                  'primary-gradient inline-flex items-center gap-3 rounded-xl px-10 py-4 font-headline font-bold text-on-primary shadow-[0_10px_20px_rgba(0,226,238,0.2)] transition-all duration-300 active:scale-95',
                  isUploading && 'opacity-70 cursor-not-allowed',
                )}
              >
                {isUploading ? (
                  <>
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-on-primary border-t-transparent" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Plus className="h-5 w-5" />
                    Select File
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {uploadError ? (
          <p className="mt-4 rounded-lg border border-red-400/40 bg-red-500/10 px-4 py-2 text-sm text-red-200">{uploadError}</p>
        ) : null}

        <div className="mt-16 grid w-full grid-cols-1 gap-4 md:grid-cols-3">
          {[
            {
              title: 'Precision Highlighting',
              desc: 'Highlight exact text directly from the PDF text layer for accurate requests.',
              icon: Edit3,
              color: 'text-secondary',
            },
            {
              title: 'Page-Aware Requests',
              desc: 'Every translation request is tied to a page number for backend matching.',
              icon: Languages,
              color: 'text-primary',
            },
            {
              title: 'Translation History',
              desc: 'Track previous translations while preserving context and source snippets.',
              icon: Archive,
              color: 'text-primary-dim',
            },
          ].map((feature) => (
            <div
              key={feature.title}
              className="rounded-xl border border-outline-variant/10 bg-surface-container-low p-6 text-left transition-colors hover:bg-surface-container-high"
            >
              <feature.icon className={cn('mb-4 h-6 w-6', feature.color)} />
              <h3 className="mb-2 font-headline text-sm font-bold text-on-surface">{feature.title}</h3>
              <p className="text-xs leading-relaxed text-on-surface-variant">{feature.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  )
}

export default function App() {
  const [view, setView] = useState<ViewState>('upload')
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [fileName, setFileName] = useState('')
  const [numPages, setNumPages] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)
  const [pageWidth, setPageWidth] = useState(720)
  const [selection, setSelection] = useState<SelectionState | null>(null)
  const [history, setHistory] = useState<TranslationHistoryItem[]>([])
  const [latestTranslation, setLatestTranslation] = useState<TranslationHistoryItem | null>(null)
  const [isTranslating, setIsTranslating] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [apiError, setApiError] = useState('')
  const [isUploadingPdf, setIsUploadingPdf] = useState(false)

  const viewerRef = useRef<HTMLDivElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    return () => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl)
      }
      abortRef.current?.abort()
    }
  }, [pdfUrl])

  useEffect(() => {
    if (view !== 'workspace') {
      return
    }

    const updateWidth = () => {
      if (!viewerRef.current) {
        return
      }

      const width = Math.max(300, Math.floor(viewerRef.current.clientWidth * 0.85))
      setPageWidth(width)
    }

    updateWidth()
    window.addEventListener('resize', updateWidth)

    return () => window.removeEventListener('resize', updateWidth)
  }, [view])

  const canTranslate = useMemo(() => !!selection?.text && !isTranslating, [selection, isTranslating])

  function resetWorkspaceForFile(file: File, objectUrl: string) {
    setPdfUrl((previousUrl) => {
      if (previousUrl) {
        URL.revokeObjectURL(previousUrl)
      }
      return objectUrl
    })
    setFileName(file.name)
    setNumPages(0)
    setCurrentPage(1)
    setSelection(null)
    setHistory([])
    setLatestTranslation(null)
    setUploadError('')
    setApiError('')
    setView('workspace')
  }

  function handleFileSelected(file: File) {
    const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')

    if (!isPdf) {
      setUploadError('Please choose a valid PDF file.')
      return
    }

    setIsUploadingPdf(true)
    setUploadError('')

    uploadPdf(file)
      .then((response) => {
        const objectUrl = URL.createObjectURL(file)
        resetWorkspaceForFile(file, objectUrl)
        setIsUploadingPdf(false)
      })
      .catch((error) => {
        setUploadError(error.message || 'Failed to upload PDF. Please try again.')
        setIsUploadingPdf(false)
      })
  }

  function handleSelectionFromPdf() {
    const browserSelection = window.getSelection()

    if (!browserSelection || browserSelection.rangeCount === 0) {
      setSelection(null)
      return
    }

    const selectedText = browserSelection.toString().replace(/\s+/g, ' ').trim()
    if (!selectedText) {
      setSelection(null)
      return
    }

    const range = browserSelection.getRangeAt(0)
    if (!viewerRef.current || !viewerRef.current.contains(range.commonAncestorContainer)) {
      return
    }

    const rect = range.getBoundingClientRect()
    const containerRect = viewerRef.current.getBoundingClientRect()

    setSelection({
      text: selectedText,
      pageNumber: currentPage,
      x: rect.left - containerRect.left + rect.width / 2,
      y: rect.top - containerRect.top - 12,
    })
  }

  async function handleTranslateSelection() {
    if (!selection || !selection.text) {
      return
    }

    setIsTranslating(true)
    setApiError('')

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const data = await requestTranslation(
        {
          text: selection.text,
          page_number: selection.pageNumber,
        },
        controller.signal,
      )

      if (data.error) {
        setApiError(data.error)
        return
      }

      if (!data.translation) {
        setApiError('The server did not return a translation.')
        return
      }

      const entry: TranslationHistoryItem = {
        id: makeLocalId(),
        original: selection.text,
        translated: data.translation,
        pageNumber: selection.pageNumber,
        timestamp: makeLocalTimestamp(),
      }

      setLatestTranslation(entry)
      setHistory((previous) => [entry, ...previous])
      setSelection(null)
      window.getSelection()?.removeAllRanges()
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        return
      }

      setApiError('Translation request failed. Please check if backend server is running.')
    } finally {
      setIsTranslating(false)
    }
  }

  function goToNextPage() {
    setCurrentPage((previous) => Math.min(previous + 1, numPages || 1))
    setSelection(null)
  }

  function goToPreviousPage() {
    setCurrentPage((previous) => Math.max(previous - 1, 1))
    setSelection(null)
  }

  return (
    <div className="min-h-screen bg-surface selection:bg-primary/30 selection:text-primary">
      <TopBar view={view} setView={setView} fileName={fileName} />

      <main className="pt-16">
        <AnimatePresence mode="wait">
          {view === 'upload' ? (
            <div key="upload">
              <UploadScreen onFileSelected={handleFileSelected} uploadError={uploadError} isUploading={isUploadingPdf} />
            </div>
          ) : (
            <motion.div
              key="workspace"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex h-[calc(100vh-64px)] w-full overflow-hidden"
            >
              <section className="pdf-canvas relative flex w-[70%] flex-col overflow-hidden border-r border-outline-variant/10 bg-surface">
                <div
                  ref={viewerRef}
                  onMouseUp={handleSelectionFromPdf}
                  className="relative flex flex-1 items-start justify-center overflow-auto p-6 md:p-12"
                >
                  <div className="relative rounded-lg border border-outline-variant/20 bg-white p-4 shadow-2xl md:p-8">
                    <Document
                      file={pdfUrl}
                      loading={<div className="p-10 text-center text-sm text-[#0e0e10]">Loading PDF...</div>}
                      onLoadSuccess={(loadedPdf) => {
                        setNumPages(loadedPdf.numPages)
                        setCurrentPage((previous) => Math.min(Math.max(previous, 1), loadedPdf.numPages))
                      }}
                      onLoadError={(error) => {
                        setApiError(`Unable to render PDF: ${String(error)}`)
                      }}
                    >
                      <Page pageNumber={currentPage} width={pageWidth} />
                    </Document>

                    {selection ? (
                      <motion.button
                        type="button"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        onClick={handleTranslateSelection}
                        disabled={!canTranslate}
                        className="glass-hud absolute z-20 inline-flex -translate-x-1/2 items-center gap-2 rounded-xl border border-primary/20 px-4 py-2 text-sm font-medium text-primary shadow-2xl disabled:cursor-not-allowed disabled:opacity-50"
                        style={{ left: `${selection.x}px`, top: `${selection.y}px` }}
                      >
                        <Languages className="h-4 w-4" />
                        {isTranslating ? 'Translating...' : 'Translate Selection'}
                      </motion.button>
                    ) : null}
                  </div>
                </div>

                <div className="glass-hud absolute bottom-8 left-1/2 z-50 flex -translate-x-1/2 items-center gap-8 rounded-full border border-outline-variant/20 px-8 py-4 shadow-2xl">
                  <button
                    type="button"
                    onClick={goToPreviousPage}
                    disabled={currentPage <= 1}
                    className="group flex items-center gap-1 text-on-surface-variant transition-all hover:text-on-surface disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <ChevronLeft className="h-5 w-5 transition-transform group-hover:-translate-x-1" />
                    <span className="text-xs font-medium">Previous</span>
                  </button>
                  <div className="flex items-center gap-2 text-xs font-bold text-primary">
                    <FileText className="h-4 w-4" />
                    Page {currentPage} of {numPages || 1}
                  </div>
                  <button
                    type="button"
                    onClick={goToNextPage}
                    disabled={numPages === 0 || currentPage >= numPages}
                    className="group flex items-center gap-1 text-on-surface-variant transition-all hover:text-on-surface disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <span className="text-xs font-medium">Next</span>
                    <ChevronRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
                  </button>
                </div>
              </section>

              <aside className="flex h-full w-[30%] flex-col border-l border-outline-variant/10 bg-surface-container-low">
                <div className="space-y-1 p-8">
                  <div className="mb-6 flex items-start justify-between">
                    <div>
                      <h2 className="font-headline text-lg font-black uppercase tracking-tight text-on-surface">
                        Translation Engine
                      </h2>
                      <span className="font-headline text-[10px] uppercase tracking-widest text-primary">V2.4 Active</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => setView('upload')}
                      className="primary-gradient rounded-md px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest text-on-primary transition-transform active:scale-95"
                    >
                      New PDF
                    </button>
                  </div>
                  {apiError ? (
                    <div className="rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2 text-xs text-red-200">
                      {apiError}
                    </div>
                  ) : null}
                </div>

                <div className="flex-1 space-y-10 overflow-y-auto px-8 pb-8">
                  <section>
                    <div className="mb-4 flex items-center justify-between">
                      <h3 className="font-headline text-[10px] uppercase tracking-widest text-on-surface-variant">
                        Latest Translation
                      </h3>
                      <span className={cn('h-1.5 w-1.5 rounded-full', isTranslating ? 'animate-pulse bg-primary' : 'bg-outline-variant')} />
                    </div>

                    {isTranslating ? (
                      <div className="space-y-4 rounded-xl border border-outline-variant/10 bg-surface-container-highest/40 p-6">
                        <div className="skeleton-shimmer h-3 w-3/4 rounded-full" />
                        <div className="skeleton-shimmer h-3 w-full rounded-full" />
                        <div className="skeleton-shimmer h-3 w-5/6 rounded-full" />
                        <div className="flex items-center justify-between pt-4">
                          <div className="flex gap-2">
                            <div className="h-2 w-8 rounded-full bg-primary/20" />
                            <div className="h-2 w-12 rounded-full bg-primary/20" />
                          </div>
                          <span className="text-[10px] font-bold text-primary">TRANSLATING...</span>
                        </div>
                      </div>
                    ) : latestTranslation ? (
                      <div className="space-y-4 rounded-xl border border-outline-variant/10 bg-surface-container-highest/40 p-6">
                        <p className="line-clamp-1 text-xs italic text-on-surface-variant">"{latestTranslation.original}"</p>
                        <p className="text-sm font-medium leading-relaxed text-on-surface">
                          {latestTranslation.translated}
                        </p>
                        <div className="flex items-center justify-between text-[10px] text-on-surface-variant">
                          <span>Page {latestTranslation.pageNumber}</span>
                          <span>{latestTranslation.timestamp}</span>
                        </div>
                      </div>
                    ) : (
                      <div className="rounded-xl border border-outline-variant/10 bg-surface-container-highest/40 p-6 text-sm text-on-surface-variant">
                        Select text in the PDF to translate it.
                      </div>
                    )}
                  </section>

                  <section className="flex-1">
                    <h3 className="mb-6 font-headline text-[10px] uppercase tracking-widest text-on-surface-variant">
                      Translation History
                    </h3>
                    <div className="space-y-4">
                      {history.length === 0 ? (
                        <div className="rounded-xl border border-outline-variant/10 bg-surface-container-highest/40 p-4 text-xs text-on-surface-variant">
                          No translations yet.
                        </div>
                      ) : (
                        history.map((item) => (
                          <div
                            key={item.id}
                            className="cursor-pointer rounded-xl border border-transparent p-4 transition-all duration-300 hover:border-outline-variant/20 hover:bg-surface-container-highest"
                          >
                            <p className="line-clamp-1 mb-2 text-xs italic text-on-surface-variant">"{item.original}"</p>
                            <p className="text-sm font-medium leading-relaxed text-on-surface">{item.translated}</p>
                            <div className="mt-3 flex items-center gap-3 text-[10px] text-on-surface-variant">
                              <span className="flex items-center gap-1">
                                <History className="h-3 w-3" />
                                {item.timestamp}
                              </span>
                              <span>Page {item.pageNumber}</span>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </section>
                </div>

                <div className="grid grid-cols-4 gap-2 border-t border-outline-variant/10 p-6">
                  {[
                    { label: 'History', icon: History, active: true },
                    { label: 'Starred', icon: Star, active: false },
                    { label: 'Drafts', icon: Edit3, active: false },
                    { label: 'Archive', icon: Archive, active: false },
                  ].map((nav) => (
                    <button
                      key={nav.label}
                      type="button"
                      className={cn(
                        'flex flex-col items-center gap-1 rounded-md p-2 transition-all duration-300',
                        nav.active
                          ? 'primary-gradient text-on-primary'
                          : 'text-on-surface-variant hover:bg-surface-container-highest',
                      )}
                    >
                      <nav.icon className={cn('h-5 w-5', nav.active ? 'fill-on-primary' : '')} />
                      <span className="font-headline text-[8px] font-bold uppercase tracking-widest">{nav.label}</span>
                    </button>
                  ))}
                </div>
              </aside>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}
