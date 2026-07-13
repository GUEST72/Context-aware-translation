import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import {
  Award,
  BookOpen,
  Clock,
  Flame,
  Languages,
  Layers,
  Settings,
  Sparkles,
  TrendingUp,
  Upload,
} from "lucide-react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
} from "recharts";
import openBook from "@/assets/open-book.jpg";
import { SiteHeader } from "@/components/site-header";
import { Button } from "@/components/button";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/dashboard")({
  head: () => ({
    meta: [
      { title: "Dashboard — LexisFlow" },
      { name: "description", content: "Your personal reading dashboard: progress, history, and analytics." },
    ],
  }),
  component: Dashboard,
});

const STATS = [
  // TODO: Replace placeholder dashboard metrics with backend-provided user analytics.
  { icon: Flame, label: "Reading streak", value: "18", unit: "days" },
  { icon: Layers, label: "Pages read", value: "2,847", unit: "" },
  { icon: Languages, label: "Passages translated", value: "412", unit: "" },
  { icon: Clock, label: "Reading time", value: "63", unit: "hrs" },
];

const ACTIVITY = [
  // TODO: Replace placeholder activity data with backend reading statistics.
  { day: "Mon", value: 34 },
  { day: "Tue", value: 52 },
  { day: "Wed", value: 41 },
  { day: "Thu", value: 78 },
  { day: "Fri", value: 63 },
  { day: "Sat", value: 96 },
  { day: "Sun", value: 71 },
];

const BOOKS = [
  // TODO: Replace placeholder library records with backend reading history.
  { title: "The Structure of Scientific Revolutions", author: "Thomas Kuhn", progress: 72, lang: "DE → EN" },
  { title: "Critique of Pure Reason", author: "Immanuel Kant", progress: 38, lang: "DE → EN" },
  { title: "Histoire de la folie", author: "Michel Foucault", progress: 91, lang: "FR → EN" },
  { title: "Ficciones", author: "Jorge Luis Borges", progress: 12, lang: "ES → EN" },
];

const PASSAGES = [
  // TODO: Replace placeholder passages with persisted translation history.
  { text: "The paradigm shift reframes normal science as a communal enterprise.", page: 84, time: "2h ago" },
  { text: "Reason, in its pure form, precedes all empirical intuition.", page: 210, time: "Yesterday" },
  { text: "Madness is measured against the silent architecture of reason.", page: 33, time: "2 days ago" },
];

const ACHIEVEMENTS = [
  // TODO: Replace placeholder achievements with backend account milestones.
  { icon: Flame, label: "Fortnight streak", earned: true },
  { icon: BookOpen, label: "First book finished", earned: true },
  { icon: Languages, label: "500 translations", earned: false },
  { icon: Award, label: "Polyglot scholar", earned: false },
];

function Dashboard() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <div className="min-h-screen parchment pb-20">
      <SiteHeader />

      <main className="mx-auto max-w-6xl px-6 pt-28">
        <div className="mb-6 rounded-2xl border border-accent/30 bg-card px-5 py-3 text-sm text-accent-foreground shadow-soft">
          Dashboard data is temporary placeholder content until profile, analytics, history, and achievements APIs are available.
        </div>

        {/* Profile header */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4 rounded-3xl border border-border bg-card p-6 shadow-soft sm:flex sm:flex-wrap sm:justify-between md:p-8"
        >
          <div className="flex min-w-0 items-center gap-4">
            <div className="grid h-16 w-16 shrink-0 place-items-center rounded-2xl [background:var(--gradient-navy)] font-serif text-2xl font-semibold text-primary-foreground">
              AL
            </div>
            <div className="min-w-0">
              <h1 className="truncate font-serif text-2xl font-semibold text-foreground md:text-3xl">
                Ada Lovelace
              </h1>
              <p className="truncate text-sm text-muted-foreground">Reader since 2024 · 14 books in library</p>
            </div>
          </div>
          <div className="flex shrink-0 gap-2">
            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4" /> Settings
            </Button>
            <Button variant="gold" size="sm" asChild>
              <Link to="/">
                <Upload className="h-4 w-4" /> Upload
              </Link>
            </Button>
          </div>
        </motion.section>

        {/* Stats */}
        <section className="mt-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
          {STATS.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: i * 0.06 }}
              className="lift rounded-2xl border border-border bg-card p-5 shadow-soft"
            >
              <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-secondary text-primary">
                <s.icon className="h-5 w-5" />
              </span>
              <p className="mt-4 font-serif text-3xl font-semibold text-foreground">
                {s.value}
                {s.unit && <span className="ml-1 text-base font-normal text-muted-foreground">{s.unit}</span>}
              </p>
              <p className="mt-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">{s.label}</p>
            </motion.div>
          ))}
        </section>

        <div className="mt-6 grid gap-6 lg:grid-cols-3">
          {/* Continue reading + library */}
          <div className="space-y-6 lg:col-span-2">
            <Card>
              <CardHeader icon={BookOpen} title="Continue reading" />
              <div className="flex flex-col gap-4 rounded-2xl border border-border bg-secondary/40 p-4 sm:flex-row sm:items-center">
                <img
                  src={openBook}
                  alt=""
                  aria-hidden
                  className="h-28 w-full rounded-xl object-cover sm:h-24 sm:w-20"
                  loading="lazy"
                  width={1200}
                  height={912}
                />
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-medium uppercase tracking-wide text-accent-foreground">DE → EN</p>
                  <h3 className="mt-1 truncate font-serif text-lg font-semibold text-foreground">
                    The Structure of Scientific Revolutions
                  </h3>
                  <p className="text-sm text-muted-foreground">Thomas Kuhn · Page 148 of 206</p>
                  <ProgressBar value={72} className="mt-3" />
                </div>
                <Button size="sm" asChild>
                  <Link to="/read">Resume</Link>
                </Button>
              </div>

              <div className="mt-6 space-y-4">
                {BOOKS.map((b) => (
                  <div key={b.title} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <h4 className="truncate text-sm font-semibold text-foreground">{b.title}</h4>
                        <span className="shrink-0 rounded-full bg-secondary px-2 py-0.5 text-[10px] font-medium text-accent-foreground">
                          {b.lang}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground">{b.author}</p>
                      <ProgressBar value={b.progress} className="mt-2" />
                    </div>
                    <span className="shrink-0 font-serif text-lg font-semibold text-foreground">{b.progress}%</span>
                  </div>
                ))}
              </div>
            </Card>

            <Card>
              <CardHeader icon={TrendingUp} title="Weekly activity" caption="Minutes read per day" />
              <div className="h-56 w-full">
                {mounted && (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={ACTIVITY} margin={{ top: 10, right: 8, left: 8, bottom: 0 }}>
                      <defs>
                        <linearGradient id="fillGold" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--gold)" stopOpacity={0.5} />
                          <stop offset="100%" stopColor="var(--gold)" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis
                        dataKey="day"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                      />
                      <Tooltip
                        cursor={{ stroke: "var(--border)" }}
                        contentStyle={{
                          background: "var(--popover)",
                          border: "1px solid var(--border)",
                          borderRadius: 12,
                          fontSize: 12,
                          color: "var(--foreground)",
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="value"
                        stroke="var(--gold)"
                        strokeWidth={2.5}
                        fill="url(#fillGold)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </div>
            </Card>
          </div>

          {/* Sidebar column */}
          <div className="space-y-6">
            <Card>
              <CardHeader icon={Award} title="Achievements" />
              <div className="grid grid-cols-2 gap-3">
                {ACHIEVEMENTS.map((a) => (
                  <div
                    key={a.label}
                    className={cn(
                      "flex flex-col items-center gap-2 rounded-2xl border p-4 text-center",
                      a.earned ? "border-accent/40 bg-secondary/50" : "border-dashed border-border opacity-60",
                    )}
                  >
                    <span
                      className={cn(
                        "flex h-10 w-10 items-center justify-center rounded-xl",
                        a.earned ? "[background:var(--gradient-gold)] text-accent-foreground" : "bg-muted text-muted-foreground",
                      )}
                    >
                      <a.icon className="h-5 w-5" />
                    </span>
                    <span className="text-xs font-medium leading-tight text-foreground">{a.label}</span>
                  </div>
                ))}
              </div>
            </Card>

            <Card>
              <CardHeader icon={Languages} title="Recently translated" />
              <div className="space-y-4">
                {PASSAGES.map((p, i) => (
                  <div key={i} className="border-l-2 border-accent/50 pl-4">
                    <p className="text-sm leading-relaxed text-foreground">"{p.text}"</p>
                    <p className="mt-1 text-[11px] text-muted-foreground">
                      Page {p.page} · {p.time}
                    </p>
                  </div>
                ))}
              </div>
            </Card>

            <Card>
              <CardHeader icon={Sparkles} title="This month" />
              <div className="space-y-4">
                <Metric label="Books started" value="4" />
                <Metric label="Books finished" value="2" />
                <Metric label="Avg. session" value="42 min" />
              </div>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.5 }}
      className="rounded-3xl border border-border bg-card p-6 shadow-soft"
    >
      {children}
    </motion.div>
  );
}

function CardHeader({
  icon: Icon,
  title,
  caption,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  caption?: string;
}) {
  return (
    <div className="mb-5 flex items-center gap-2.5">
      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-secondary text-primary">
        <Icon className="h-4.5 w-4.5" />
      </span>
      <div>
        <h2 className="font-serif text-lg font-semibold text-foreground">{title}</h2>
        {caption && <p className="text-xs text-muted-foreground">{caption}</p>}
      </div>
    </div>
  );
}

function ProgressBar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full bg-muted", className)}>
      <motion.div
        initial={{ width: 0 }}
        whileInView={{ width: `${value}%` }}
        viewport={{ once: true }}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        className="h-full rounded-full [background:var(--gradient-gold)]"
      />
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="font-serif text-lg font-semibold text-foreground">{value}</span>
    </div>
  );
}
