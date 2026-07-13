import { Link } from "@tanstack/react-router";
import { motion } from "motion/react";
import { BookOpen, Quote } from "lucide-react";
import type { ReactNode } from "react";
import heroLibrary from "@/assets/hero-library.jpg";
import { DustParticles } from "@/components/atmosphere";

export function AuthShell({
  title,
  subtitle,
  children,
  footer,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
  footer: ReactNode;
}) {
  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      {/* Visual side */}
      <div className="relative hidden overflow-hidden lg:block">
        <img src={heroLibrary} alt="" aria-hidden className="h-full w-full object-cover" />
        <div className="absolute inset-0 [background:var(--gradient-navy)] opacity-70" />
        <DustParticles count={22} />
        <div className="absolute inset-0 flex flex-col justify-between p-12">
          <Link to="/" className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-cream/15 backdrop-blur-sm">
              <BookOpen className="h-4.5 w-4.5 text-cream" />
            </span>
            <span className="font-serif text-xl font-semibold text-cream">LexisFlow</span>
          </Link>
          <div>
            <Quote className="mb-4 h-8 w-8 text-gold" />
            <p className="max-w-md font-serif text-2xl font-medium italic leading-snug text-cream">
              “A book is a garden carried in the pocket — LexisFlow lets you read
              every garden, in every tongue.”
            </p>
            <p className="mt-4 text-sm text-cream/70">The LexisFlow reading room</p>
          </div>
        </div>
      </div>

      {/* Form side */}
      <div className="flex items-center justify-center bg-background px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="w-full max-w-md"
        >
          <Link to="/" className="mb-8 flex items-center gap-2.5 lg:hidden">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl [background:var(--gradient-navy)]">
              <BookOpen className="h-4.5 w-4.5 text-primary-foreground" />
            </span>
            <span className="font-serif text-xl font-semibold text-foreground">LexisFlow</span>
          </Link>

          <h1 className="font-serif text-3xl font-semibold tracking-tight text-foreground">{title}</h1>
          <p className="mt-2 text-sm text-muted-foreground">{subtitle}</p>

          <div className="mt-8">{children}</div>

          <p className="mt-8 text-center text-sm text-muted-foreground">{footer}</p>
        </motion.div>
      </div>
    </div>
  );
}

export function Field({
  label,
  ...props
}: { label: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium text-foreground">{label}</span>
      <input
        {...props}
        className="h-11 w-full rounded-xl border border-input bg-card px-4 text-sm text-foreground outline-none transition-all placeholder:text-muted-foreground focus:border-ring focus:ring-2 focus:ring-ring/30"
      />
    </label>
  );
}

export function GoogleButton({ label }: { label: string }) {
  return (
    <button
      type="button"
      className="flex h-11 w-full items-center justify-center gap-2.5 rounded-full border border-border bg-card text-sm font-medium text-foreground transition-colors hover:bg-secondary active:scale-[0.98]"
    >
      <svg className="h-4 w-4" viewBox="0 0 24 24" aria-hidden>
        <path fill="#EA4335" d="M12 10.2v3.9h5.5c-.24 1.4-1.7 4.1-5.5 4.1-3.3 0-6-2.7-6-6.1s2.7-6.1 6-6.1c1.9 0 3.1.8 3.8 1.5l2.6-2.5C16.9 3 14.7 2 12 2 6.9 2 2.8 6.1 2.8 11.2S6.9 20.4 12 20.4c5.3 0 8.8-3.7 8.8-9 0-.6-.06-1-.15-1.4H12z" />
      </svg>
      {label}
    </button>
  );
}
