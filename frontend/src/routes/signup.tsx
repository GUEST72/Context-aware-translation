import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { AuthShell, Field, GoogleButton } from "@/components/auth-shell";
import { Button } from "@/components/button";

export const Route = createFileRoute("/signup")({
  head: () => ({
    meta: [
      { title: "Sign up — LexisFlow" },
      { name: "description", content: "Create your LexisFlow account and start translating academic books." },
    ],
  }),
  component: SignupPage,
});

function SignupPage() {
  const [notice, setNotice] = useState("");

  return (
    <AuthShell
      title="Create your account"
      subtitle="Join a reading room built for scholars and researchers."
      footer={
        <>
          Already have an account?{" "}
          <Link to="/login" className="font-medium text-foreground underline-offset-4 hover:underline">
            Log in
          </Link>
        </>
      }
    >
      <form
        className="space-y-5"
        onSubmit={(e) => {
          e.preventDefault();
          // TODO: Connect to the real sign-up endpoint when the backend contract exists.
          setNotice("Sign up is a frontend placeholder until account APIs are available.");
        }}
      >
        <Field label="Full name" type="text" placeholder="Ada Lovelace" required />
        <Field label="Email" type="email" placeholder="you@university.edu" required />
        <Field label="Password" type="password" placeholder="••••••••" required />
        <Field label="Confirm password" type="password" placeholder="••••••••" required />

        <Button type="submit" size="lg" className="w-full">
          Create account
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

      <p className="mt-6 text-center text-xs text-muted-foreground">
        By creating an account you agree to our Terms & Privacy Policy.
      </p>
    </AuthShell>
  );
}
