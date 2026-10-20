import { Compass, Loader2, TriangleAlert } from "lucide-react";
import { Link, useRouteError } from "react-router";
import { EmptyState } from "@/components/common/States";
import { Button } from "@/components/ui/button";

export function NotFoundPage() {
  return (
    <EmptyState
      icon={Compass}
      title="Page not found"
      description="That address doesn't match any page in the portal."
      action={
        <Button asChild>
          <Link to="/">Go to Diagnose</Link>
        </Button>
      }
    />
  );
}

/** Shown when a page crashes or fails to load (e.g. a lazy chunk after a redeploy). */
export function RouteErrorPage() {
  const error = useRouteError();
  return (
    <div className="flex min-h-dvh items-center justify-center p-6">
      <EmptyState
        icon={TriangleAlert}
        title="This page couldn't be shown"
        description={error?.message || "An unexpected error happened. Reloading usually fixes it."}
        action={<Button onClick={() => window.location.reload()}>Reload</Button>}
      />
    </div>
  );
}

/** Brief spinner while a page's code loads. */
export function PageLoading() {
  return (
    <div className="flex min-h-dvh items-center justify-center" role="status">
      <Loader2 className="size-6 animate-spin text-primary" aria-hidden="true" />
      <span className="sr-only">Loading…</span>
    </div>
  );
}
