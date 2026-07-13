import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { motion, useScroll, useTransform } from "motion/react";
import { useRef, useState } from "react";
import {
  ArrowRight,
  BookMarked,
  ChevronDown,
  FileText,
  Languages,
  Library,
  Sparkles,
  Upload,
} from "lucide-react";
import heroLibrary from "@/assets/hero-library.jpg";
import openBook from "@/assets/open-book.jpg";
import { SiteHeader } from "@/components/site-header";
import { DustParticles, LightRays } from "@/components/atmosphere";
import { Button } from "@/components/button";
import { useLibrary } from "@/lib/library-store";
import { uploadPdf } from "@/lib/translationApi";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  component: Landing,
});

const STORY = [
  {
    icon: Library,
    title: "A library that thinks",
    body: "Bring any academic PDF into a calm, focused reading room designed for deep scholarly work.",
  },
  {
    icon: Languages,
    title: "Translation with fidelity",
    body: "Highlight a passage and receive a translation that preserves terminology, structure, and intent.",
  },
  {
    icon: BookMarked,
    title: "Knowledge that stays",
    body: "Every rendered passage is kept beside your reading, building a personal archive of understanding.",
  },
];

function Landing() {
  const navigate = useNavigate();
  const { setDocument } = useLibrary();
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const heroRef = useRef<HTMLDivElement | null>(null);
  const { scrollYProgress: heroProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  });
  const heroScale = useTransform(heroProgress, [0, 1], [1, 1.15]);
  const heroTextY = useTransform(heroProgress, [0, 1], ["0%", "40%"]);
  const heroTextOpacity = useTransform(heroProgress, [0, 0.7], [1, 0]);

  const splitRef = useRef<HTMLDivElement | null>(null);
  const { scrollYProgress: splitProgress } = useScroll({
      target: splitRef,
      offset: ["start start", "end start"],
  });
  const leftX = useTransform(
      splitProgress,
      [0.04, 0.48, 0.80],
      ["0%", "-102%", "-102%"]
  );

  const rightX = useTransform(
      splitProgress,
      [0.04, 0.48, 0.80],
      ["0%", "102%", "102%"]
  );
  const panelsOpacity = 1;
  
  const uploadOpacity = useTransform(
    splitProgress,
    [0.42, 0.55, 0.80, 1],
    [0, 1, 1, 1]
  );

  const uploadScale = useTransform(
    splitProgress,
    [0.42, 0.55, 0.80, 1],
    [0.95, 1, 1, 1]
  ); 
   const seamGlow = useTransform(splitProgress, [0.04, 0.36], [0, 1]);


  async function handleFile(file: File | undefined) {
    if (!file) return;
    const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
    if (!isPdf) {
      setError("Please choose a valid PDF file.");
      return;
    }
    setError("");
    setIsUploading(true);
    try {
      await uploadPdf(file);
      setDocument(file);
      navigate({ to: "/read" });
    } catch (e) {
      setError((e as Error).message || "Upload failed. Please try again.");
      setIsUploading(false);
    }
  }

  return (
    <div className="relative min-h-screen bg-background">
      <SiteHeader overlay />

      {/* HERO */}
      <section ref={heroRef} className="relative h-screen overflow-hidden">
        <motion.div style={{ scale: heroScale }} className="absolute inset-0">
          <img
            src={heroLibrary}
            alt="A magnificent historic library reading room bathed in warm golden light"
            className="h-full w-full object-cover"
            width={1920}
            height={1280}
          />
          <div className="absolute inset-0 bg-gradient-to-b from-navy/70 via-navy/40 to-background" />
          <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent" />
        </motion.div>

        <LightRays />
        <DustParticles count={30} />

        <motion.div
          style={{ y: heroTextY, opacity: heroTextOpacity }}
          className="relative z-10 mx-auto flex h-full max-w-4xl flex-col items-center justify-center px-6 text-center"
        >
          <motion.span
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.7 }}
            className="mb-6 inline-flex items-center gap-2 rounded-full border border-gold/40 bg-navy/30 px-4 py-1.5 text-xs font-medium tracking-wide text-cream backdrop-blur-sm"
          >
            <Sparkles className="h-3.5 w-3.5 text-gold" />
            Intelligent translation for scholars
          </motion.span>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.9 }}
            className="font-serif text-5xl font-semibold leading-[1.05] tracking-tight text-cream md:text-7xl"
          >
            Where great books
            <br />
            <span className="italic text-gradient-gold">speak every language.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.9 }}
            className="mx-auto mt-6 max-w-xl text-lg leading-relaxed text-cream/80"
          >
            Translate academic books intelligently while preserving terminology,
            structure, and meaning — inside a reading room built for research.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.9 }}
            className="mt-9 flex flex-col items-center gap-3 sm:flex-row"
          >
            <Button
              variant="gold"
              size="lg"
              onClick={() => document.getElementById("upload")?.scrollIntoView({ behavior: "smooth" })}
            >
              Begin translating
              <ArrowRight className="h-4 w-4" />
            </Button>
          </motion.div>
        </motion.div>

        <div className="absolute bottom-8 left-1/2 z-10 -translate-x-1/2">
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
            className="flex flex-col items-center gap-2 text-cream/70"
          >
            <span className="text-[11px] uppercase tracking-[0.25em]">Scroll</span>
            <ChevronDown className="h-5 w-5" />
          </motion.div>
        </div>
      </section>

      {/* STORY */}
      <section className="relative mx-auto max-w-6xl px-6 py-28">
        <div className="mx-auto max-w-2xl text-center">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.25em] text-accent-foreground">
            The reading room
          </p>
          <h2 className="font-serif text-4xl font-semibold tracking-tight text-foreground md:text-5xl">
            Scholarship, quietly amplified
          </h2>
          <p className="mx-auto mt-5 max-w-xl text-base leading-relaxed text-muted-foreground">
            LexisFlow treats every document with the care of a rare-books
            collection — and the intelligence of a tireless research assistant.
          </p>
        </div>

        <div className="mt-16 grid gap-6 md:grid-cols-3">
          {STORY.map((item, i) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 28 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.7, delay: i * 0.1 }}
              className="lift rounded-3xl border border-border bg-card p-8 shadow-soft"
            >
              <span className="mb-6 flex h-12 w-12 items-center justify-center rounded-2xl bg-secondary text-primary">
                <item.icon className="h-6 w-6" />
              </span>
              <h3 className="font-serif text-xl font-semibold text-foreground">{item.title}</h3>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{item.body}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* HERO SPLIT + UPLOAD REVEAL */}
      <section ref={splitRef} id="upload" className="relative h-[400vh]">
          <div className="absolute inset-0 parchment" />
          <div className="sticky top-0 h-screen overflow-hidden">
          
          {/* Left door */}
          <motion.div
            style={{ x: leftX, opacity: panelsOpacity }}
            className="absolute inset-y-0 left-0 z-10 w-1/2 overflow-hidden border-r border-gold/30"
          >
            <img src={heroLibrary} alt="" aria-hidden className="h-full w-[200%] max-w-none object-cover" />
            <div className="absolute inset-0 bg-navy/55" />
            <div className="absolute inset-y-0 right-0 flex w-full items-center justify-end pr-10 text-right">
              <div>
                <p className="font-serif text-3xl font-semibold text-cream md:text-5xl">Open the</p>
                <p className="font-serif text-3xl font-semibold italic text-gradient-gold md:text-5xl">archive</p>
              </div>
            </div>
          </motion.div>

          {/* Right door */}
          <motion.div
            style={{ x: rightX, opacity: panelsOpacity }}
            className="absolute inset-y-0 right-0 z-10 w-1/2 overflow-hidden border-l border-gold/30"
          >
            <img
              src={heroLibrary}
              alt=""
              aria-hidden
              className="absolute right-0 h-full w-[200%] max-w-none object-cover"
              style={{ objectPosition: "right" }}
            />
            <div className="absolute inset-0 bg-navy/55" />
            <div className="absolute inset-y-0 left-0 flex w-full items-center pl-10">
              <div>
                <p className="font-serif text-3xl font-semibold text-cream md:text-5xl">and let it</p>
                <p className="font-serif text-3xl font-semibold italic text-gradient-gold md:text-5xl">translate</p>
              </div>
            </div>
          </motion.div>
          {/* Upload card behind the doors */}
          
            <motion.div
              style={{
                scale: uploadScale,
                opacity: uploadOpacity,
                willChange: "transform, opacity",
              }}
              className="absolute inset-0 z-[100] flex items-center justify-center px-6 pointer-events-auto"
            > 
            <UploadCard
              isUploading={isUploading}
              isDragging={isDragging}
              error={error}
              onBrowse={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                handleFile(e.dataTransfer.files?.[0]);
              }}
            />
            <input
              ref={inputRef}
              type="file"
              accept="application/pdf"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </motion.div>

          {/* Seam glow between the doors */}
          <motion.div
            style={{ opacity: seamGlow }}
            className="absolute inset-y-0 left-1/2 z-20 w-px -translate-x-1/2 bg-gradient-to-b from-transparent via-gold to-transparent blur-[1px]"
          />
        </div>
      </section>

      {/* CLOSING */}
      <footer className="border-t border-border bg-card">
        <div className="mx-auto grid max-w-6xl gap-10 px-6 py-16 md:grid-cols-[1.5fr_1fr_1fr]">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl [background:var(--gradient-navy)]">
                <BookMarked className="h-4.5 w-4.5 text-primary-foreground" />
              </span>
              <span className="font-serif text-xl font-semibold text-foreground">LexisFlow</span>
            </div>
            <p className="mt-4 max-w-sm text-sm leading-relaxed text-muted-foreground">
              An AI-powered reading room for translating and studying academic
              books — with terminology and context preserved.
            </p>
          </div>
          <FooterCol title="Product" items={["Reader", "Dashboard", "Library", "Pricing"]} />
          <FooterCol title="Company" items={["About", "Research", "Privacy", "Contact"]} />
        </div>
        <div className="border-t border-border px-6 py-6 text-center text-xs text-muted-foreground">
          © {new Date().getFullYear()} LexisFlow. Built for scholars.
        </div>
      </footer>
    </div>
  );
}

function FooterCol({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <p className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-foreground">{title}</p>
      <ul className="space-y-2.5">
        {items.map((i) => (
          <li key={i}>
            <span className="cursor-pointer text-sm text-muted-foreground transition-colors hover:text-foreground">
              {i}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function UploadCard({
  isUploading,
  isDragging,
  error,
  onBrowse,
  onDragOver,
  onDragLeave,
  onDrop,
}: {
  isUploading: boolean;
  isDragging: boolean;
  error: string;
  onBrowse: () => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: () => void;
  onDrop: (e: React.DragEvent) => void;
}) {
  return (
    <div className="rounded-[2rem] border border-border bg-card p-2 shadow-book">
      <div
        onClick={onBrowse}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={cn(
          "group cursor-pointer rounded-[1.7rem] border-2 border-dashed p-10 text-center transition-all duration-300 md:p-14",
          isDragging ? "border-accent bg-secondary" : "border-border bg-background hover:bg-secondary/50",
        )}
      >
        <div className="relative mx-auto mb-6 h-24 w-20">
          <img
            src={openBook}
            alt=""
            aria-hidden
            className="h-full w-full rounded-lg object-cover shadow-soft transition-transform duration-500 group-hover:-translate-y-1 group-hover:rotate-2"
            width={1200}
            height={912}
          />
          <span className="absolute -bottom-2 -right-2 flex h-9 w-9 items-center justify-center rounded-xl [background:var(--gradient-gold)] shadow-soft">
            {isUploading ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-accent-foreground border-t-transparent" />
            ) : (
              <Upload className="h-4 w-4 text-accent-foreground" />
            )}
          </span>
        </div>

        <h3 className="font-serif text-2xl font-semibold text-foreground">
          {isUploading ? "Preparing your book…" : "Upload an academic PDF"}
        </h3>
        <p className="mx-auto mt-2 max-w-sm text-sm text-muted-foreground">
          Drag & drop your document here, or browse to select. We open it in a
          distraction-free reading room.
        </p>

        <Button variant="primary" size="lg" className="mt-7" disabled={isUploading}>
          <FileText className="h-4 w-4" />
          {isUploading ? "Processing…" : "Browse files"}
        </Button>

        <p className="mt-4 text-xs uppercase tracking-[0.2em] text-muted-foreground">
          PDF · up to 100 MB
        </p>
      </div>

      {error && (
        <p className="mx-2 mb-2 mt-1 rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-2.5 text-sm text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}
