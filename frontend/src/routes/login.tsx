import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { AuthShell, Field, GoogleButton } from "@/components/auth-shell";
import { Button } from "@/components/button";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Log in — LexisFlow" },
      { name: "description", content: "Log in to your LexisFlow reading room." },
    ],
  }),
  component: LoginPage,
});

function LoginPage() {
  const [notice, setNotice] = useState("");

  return (
    <AuthShell
      title="Welcome back"
      subtitle="Return to your reading room and continue where you left off."
      footer={
        <>
          New to LexisFlow?{" "}
          <Link to="/signup" className="font-medium text-foreground underline-offset-4 hover:underline">
            Create an account
          </Link>
        </>
      }
    >
      <form
        className="space-y-5"
        onSubmit={(e) => {
          e.preventDefault();
          // TODO: Connect to the real authentication endpoint when the backend contract exists.
          setNotice("Login is a frontend placeholder until authentication APIs are available.");
        }}
      >
        <Field label="Email" type="email" placeholder="you@university.edu" required />
        <Field label="Password" type="password" placeholder="••••••••" required />

        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            <input type="checkbox" className="h-4 w-4 rounded border-input accent-[var(--primary)]" />
            Remember me
          </label>
          <button type="button" className="text-sm font-medium text-accent-foreground hover:underline">
            Forgot password?
          </button>
        </div>

        <Button type="submit" size="lg" className="w-full">
          Log in
        </Button>
        {notice && (
          <p className="rounded-xl border border-accent/30 bg-secondary/60 px-4 py-3 text-sm text-accent-foreground">
            {notice}
          </p>
        )}
      </form>

      <div className="my-6 flex items-center gap-4">
        <span className="h-px flex-1 bg-border" />
        <span className="text-xs uppercase tracking-widest text-muted-foreground">or</span>
        <span className="h-px flex-1 bg-border" />
      </div>

      <GoogleButton label="Continue with Google" />
    </AuthShell>
  );
}
