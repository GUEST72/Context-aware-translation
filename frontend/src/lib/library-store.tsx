import { createContext, useContext, useState, type ReactNode } from "react";

export interface TranslationHistoryItem {
  id: string;
  original: string;
  translated: string;
  pageNumber: number;
  timestamp: string;
  starred?: boolean;
}

interface LibraryState {
  file: File | null;
  pdfUrl: string | null;
  fileName: string;
  history: TranslationHistoryItem[];
  setDocument: (file: File) => void;
  clearDocument: () => void;
  addTranslation: (item: TranslationHistoryItem) => void;
  toggleStar: (id: string) => void;
}

const LibraryContext = createContext<LibraryState | null>(null);

export function LibraryProvider({ children }: { children: ReactNode }) {
  const [file, setFile] = useState<File | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const [history, setHistory] = useState<TranslationHistoryItem[]>([]);

  function setDocument(nextFile: File) {
    setPdfUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(nextFile);
    });
    setFile(nextFile);
    setFileName(nextFile.name);
    setHistory([]);
  }

  function clearDocument() {
    setPdfUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setFile(null);
    setFileName("");
    setHistory([]);
  }

  function addTranslation(item: TranslationHistoryItem) {
    setHistory((prev) => [item, ...prev]);
  }

  function toggleStar(id: string) {
    setHistory((prev) =>
      prev.map((h) => (h.id === id ? { ...h, starred: !h.starred } : h)),
    );
  }

  return (
    <LibraryContext.Provider
      value={{ file, pdfUrl, fileName, history, setDocument, clearDocument, addTranslation, toggleStar }}
    >
      {children}
    </LibraryContext.Provider>
  );
}

export function useLibrary() {
  const ctx = useContext(LibraryContext);
  if (!ctx) throw new Error("useLibrary must be used within LibraryProvider");
  return ctx;
}

export function makeLocalId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function makeLocalTimestamp() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
