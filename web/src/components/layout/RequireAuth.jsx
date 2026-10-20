import { Navigate, Outlet, useLocation } from "react-router";
import { Logo } from "@/components/layout/Logo";
import { useAuth } from "@/providers/contexts";

/** Signed-in routes; everyone else goes to /login and comes back afterwards. */
export function RequireAuth() {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="flex min-h-dvh flex-col items-center justify-center gap-4" role="status">
        <Logo compact className="animate-pulse" />
        <span className="text-sm text-muted-foreground">Checking your session…</span>
      </div>
    );
  }
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
  return <Outlet />;
}
