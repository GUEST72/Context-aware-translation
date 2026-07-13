import { Link, useRouterState } from "@tanstack/react-router";
import { BookOpen, Menu, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/button";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Home" },
  { to: "/read", label: "Reader" },
  { to: "/dashboard", label: "Dashboard" },
];

export function SiteHeader({ overlay = false }: { overlay?: boolean }) {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const solid = !overlay || scrolled;

  return (
    <header className="fixed inset-x-0 top-0 z-50 flex justify-center px-4 pt-4">
      <div
        className={cn(
          "flex w-full max-w-6xl items-center justify-between rounded-full px-4 py-2.5 transition-all duration-500",
          solid ? "glass-panel shadow-soft" : "border border-transparent",
        )}
      >
        <Link to="/" className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl [background:var(--gradient-navy)] shadow-soft">
            <BookOpen className="h-4.5 w-4.5 text-primary-foreground" />
          </span>
          <span
            className={cn(
              "font-serif text-xl font-semibold tracking-tight transition-colors",
              solid ? "text-foreground" : "text-cream",
            )}
          >
            LexisFlow
          </span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {NAV.map((item) => {
            const active = pathname === item.to;
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  "rounded-full px-4 py-2 text-sm font-medium tracking-tight transition-colors",
                  active
                    ? "bg-secondary text-foreground"
                    : solid
                      ? "text-muted-foreground hover:text-foreground"
                      : "text-cream/80 hover:text-cream",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="hidden items-center gap-2 md:flex">
          <Button
            asChild
            variant={solid ? "ghost" : "outline"}
            size="sm"
            className={cn(!solid && "border-cream/40 bg-cream/5 text-cream hover:bg-cream/15 hover:text-cream")}
          >
            <Link to="/login">Log in</Link>
          </Button>
          <Button asChild variant="gold" size="sm">
            <Link to="/signup">Sign up</Link>
          </Button>
        </div>

        <button
          type="button"
          className={cn("rounded-full p-2 md:hidden", solid ? "text-foreground" : "text-cream")}
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle menu"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {open && (
        <div className="absolute left-4 right-4 top-20 z-50 rounded-3xl glass-panel p-4 shadow-elegant md:hidden">
          <nav className="flex flex-col gap-1">
            {NAV.map((item) => (
              <Link
                key={item.to}
                to={item.to}
                onClick={() => setOpen(false)}
                className="rounded-xl px-4 py-3 text-sm font-medium text-foreground hover:bg-secondary"
              >
                {item.label}
              </Link>
            ))}
            <div className="mt-2 grid grid-cols-2 gap-2">
              <Button asChild variant="outline" size="sm">
                <Link to="/login" onClick={() => setOpen(false)}>Log in</Link>
              </Button>
              <Button asChild variant="gold" size="sm">
                <Link to="/signup" onClick={() => setOpen(false)}>Sign up</Link>
              </Button>
            </div>
          </nav>
        </div>
      )}
    </header>
  );
}
