import { motion } from "framer-motion";
import { Eye, EyeOff, Loader2, ScanSearch, ShieldCheck, Sparkles, Stethoscope } from "lucide-react";
import { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router";
import { ClassLabel, StatusBadge } from "@/components/common/Badges";
import { Disclaimer } from "@/components/common/Disclaimer";
import { ConfidenceGauge } from "@/components/diagnose/ConfidenceGauge";
import { Logo } from "@/components/layout/Logo";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAuth } from "@/providers/contexts";

const FEATURES = [
  {
    icon: ScanSearch,
    title: "Calibrated confidence",
    text: "Probabilities are rescaled so that 90% means right about 9 times in 10 on the test set.",
  },
  {
    icon: Stethoscope,
    title: "Uncertain cases are referred",
    text: "Anything below 90% confidence is flagged for clinician review instead of being guessed.",
  },
  {
    icon: Sparkles,
    title: "Explained, not just scored",
    text: "A heatmap shows where the model looked, and a checked plain-language report explains the result.",
  },
];

function GoogleIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" aria-hidden="true">
      <path fill="#4285F4" d="M23.5 12.3c0-.8-.1-1.6-.2-2.3H12v4.4h6.5a5.5 5.5 0 0 1-2.4 3.6v3h3.9c2.2-2.1 3.5-5.1 3.5-8.7Z" />
      <path fill="#34A853" d="M12 24c3.2 0 6-1.1 8-2.9l-3.9-3c-1.1.7-2.5 1.2-4.1 1.2-3.1 0-5.8-2.1-6.7-5H1.3v3.1A12 12 0 0 0 12 24Z" />
      <path fill="#FBBC05" d="M5.3 14.3a7.2 7.2 0 0 1 0-4.6V6.6h-4a12 12 0 0 0 0 10.8l4-3.1Z" />
      <path fill="#EA4335" d="M12 4.8c1.8 0 3.3.6 4.6 1.8l3.4-3.4A12 12 0 0 0 1.3 6.6l4 3.1c.9-2.9 3.6-4.9 6.7-4.9Z" />
    </svg>
  );
}

function AuthForm({ mode }) {
  const { signIn, signUp } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const signingUp = mode === "signup";

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    setNotice(null);
    setBusy(true);
    try {
      if (signingUp) {
        const { needsConfirmation } = await signUp(email, password);
        if (needsConfirmation) {
          setNotice("Check your inbox: we sent a link to confirm your email, then sign in.");
          return;
        }
      } else {
        await signIn(email, password);
      }
      navigate(location.state?.from || "/", { replace: true });
    } catch (err) {
      setError(err.message || "Sign-in failed. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  const errorId = `${mode}-error`;
  return (
    <form onSubmit={onSubmit} className="space-y-4" noValidate={false}>
      <div className="space-y-2">
        <Label htmlFor={`${mode}-email`}>Email</Label>
        <Input
          id={`${mode}-email`}
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          aria-describedby={error ? errorId : undefined}
          placeholder="you@hospital.org"
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor={`${mode}-password`}>Password</Label>
        <div className="relative">
          <Input
            id={`${mode}-password`}
            type={showPassword ? "text" : "password"}
            autoComplete={signingUp ? "new-password" : "current-password"}
            required
            minLength={signingUp ? 8 : undefined}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            aria-describedby={[signingUp ? `${mode}-hint` : null, error ? errorId : null].filter(Boolean).join(" ") || undefined}
            className="pr-10"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="absolute top-1/2 right-1 -translate-y-1/2 text-muted-foreground"
            onClick={() => setShowPassword((v) => !v)}
            aria-label={showPassword ? "Hide password" : "Show password"}
            aria-pressed={showPassword}
          >
            {showPassword ? <EyeOff aria-hidden="true" /> : <Eye aria-hidden="true" />}
          </Button>
        </div>
        {signingUp && (
          <p id={`${mode}-hint`} className="text-xs text-muted-foreground">
            At least 8 characters.
          </p>
        )}
      </div>
      {error && (
        <p id={errorId} role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {notice && (
        <p role="status" className="rounded-md bg-accent px-3 py-2 text-sm text-accent-foreground">
          {notice}
        </p>
      )}
      <Button type="submit" className="h-10 w-full" disabled={busy}>
        {busy && <Loader2 className="animate-spin" aria-hidden="true" />}
        {signingUp ? "Create account" : "Sign in"}
      </Button>
    </form>
  );
}

function AuthCard() {
  const { configured, signInWithGoogle } = useAuth();
  const [googleError, setGoogleError] = useState(null);
  const [googleBusy, setGoogleBusy] = useState(false);

  if (!configured) {
    return (
      <Alert>
        <ShieldCheck aria-hidden="true" />
        <AlertTitle>Sign-in isn&apos;t configured</AlertTitle>
        <AlertDescription>
          Copy <code>web/.env.example</code> to <code>web/.env.local</code>, set your Supabase URL and anon key, and
          restart <code>npm run dev</code>.
        </AlertDescription>
      </Alert>
    );
  }

  async function google() {
    setGoogleError(null);
    setGoogleBusy(true);
    try {
      await signInWithGoogle(); // the browser leaves for Google and comes back signed in
    } catch (err) {
      setGoogleError(err.message || "Google sign-in failed.");
      setGoogleBusy(false);
    }
  }

  return (
    <Card className="w-full shadow-lg shadow-primary/5">
      <CardHeader>
        <CardTitle className="text-xl">Welcome</CardTitle>
        <CardDescription>Sign in to analyse chest X-rays and review your history.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <Button variant="outline" className="h-10 w-full" onClick={google} disabled={googleBusy}>
          {googleBusy ? <Loader2 className="animate-spin" aria-hidden="true" /> : <GoogleIcon />}
          Continue with Google
        </Button>
        {googleError && (
          <p role="alert" className="text-sm text-destructive">
            {googleError}
          </p>
        )}
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <Separator className="flex-1" />
          or with email
          <Separator className="flex-1" />
        </div>
        <Tabs defaultValue="signin">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="signin">Sign in</TabsTrigger>
            <TabsTrigger value="signup">Create account</TabsTrigger>
          </TabsList>
          <TabsContent value="signin" className="pt-4">
            <AuthForm mode="signin" />
          </TabsContent>
          <TabsContent value="signup" className="pt-4">
            <AuthForm mode="signup" />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

/** Example output in the hero, so first-time visitors see what they'll get. */
function SampleResult() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.15, ease: "easeOut" }}
      className="relative mx-auto w-full max-w-sm rounded-2xl border bg-card p-5 shadow-xl shadow-primary/10"
      aria-label="Example result"
    >
      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Example result</span>
        <StatusBadge status="confident" />
      </div>
      <ConfidenceGauge value={0.973} status="confident" size="sm" />
      <div className="mt-3 flex items-center justify-between rounded-lg bg-muted px-3 py-2 text-sm">
        <span className="text-muted-foreground">Most likely</span>
        <ClassLabel name="Normal" className="font-medium" />
      </div>
    </motion.div>
  );
}

export function LoginPage() {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (!loading && user) return <Navigate to={location.state?.from || "/"} replace />;

  return (
    <div className="relative min-h-dvh overflow-hidden bg-background">
      {/* Soft background wash */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(60rem_40rem_at_10%_-10%,color-mix(in_oklab,var(--primary)_14%,transparent),transparent),radial-gradient(40rem_30rem_at_100%_100%,color-mix(in_oklab,var(--chart-normal)_10%,transparent),transparent)]"
      />
      <header className="relative mx-auto flex max-w-6xl items-center justify-between px-4 py-5 sm:px-6 lg:px-10">
        <Logo />
        <ThemeToggle />
      </header>

      {/* Phone order: headline, sign-in, features. Desktop: headline + features left, sign-in right. */}
      <main className="relative mx-auto grid max-w-6xl gap-8 px-4 pt-4 pb-12 sm:px-6 lg:grid-cols-[1.15fr_1fr] lg:grid-rows-[auto_1fr] lg:gap-x-16 lg:gap-y-8 lg:px-10 lg:pt-12">
        <section aria-labelledby="hero-title" className="lg:col-start-1 lg:row-start-1">
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }}>
            <p className="mb-4 inline-flex items-center gap-2 rounded-full border bg-card px-3 py-1 text-xs font-medium text-muted-foreground">
              <span className="size-1.5 rounded-full bg-success" aria-hidden="true" /> Normal · Pneumonia · Tuberculosis
            </p>
            <h1 id="hero-title" className="text-3xl leading-tight font-semibold tracking-tight text-balance sm:text-4xl lg:text-5xl">
              Chest X-ray screening that knows when to <span className="text-primary">ask a clinician</span>
            </h1>
            <p className="mt-4 max-w-xl text-base text-muted-foreground sm:text-lg">
              Upload an X-ray and get a calibrated result in seconds, see where the model looked, and download a
              plain-language report.
            </p>
          </motion.div>
        </section>

        <section aria-label="What you get" className="lg:col-start-1 lg:row-start-2">
          <ul className="grid gap-4 sm:grid-cols-3 lg:grid-cols-1 lg:gap-5">
            {FEATURES.map(({ icon: Icon, title, text }, i) => (
              <motion.li
                key={title}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, delay: 0.1 + i * 0.08 }}
                className="rounded-xl border bg-card/70 p-4 lg:flex lg:gap-4 lg:border-0 lg:bg-transparent lg:p-0"
              >
                <span className="mb-2 flex size-9 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-foreground lg:mb-0">
                  <Icon className="size-5" aria-hidden="true" />
                </span>
                <span className="block">
                  <h2 className="text-sm font-semibold">{title}</h2>
                  <p className="mt-1 text-sm text-muted-foreground">{text}</p>
                </span>
              </motion.li>
            ))}
          </ul>
          <div className="mt-10 hidden lg:block">
            <SampleResult />
          </div>
        </section>

        <section aria-label="Sign in" className="flex flex-col gap-6 lg:col-start-2 lg:row-span-2 lg:row-start-1 lg:justify-center">
          <AuthCard />
          <p className="text-xs text-muted-foreground">
            97.6% accuracy on the held-out test set. Each disease comes from a different public dataset, so the model may
            partly recognise the source; expect lower accuracy on other hospitals&apos; images.
          </p>
          <Disclaimer />
        </section>
      </main>
    </div>
  );
}
