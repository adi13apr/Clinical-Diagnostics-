import { AnimatePresence, motion } from "framer-motion";
import { NavLink, Outlet, useLocation } from "react-router";
import { Disclaimer } from "@/components/common/Disclaimer";
import { HealthIndicator } from "@/components/layout/HealthIndicator";
import { Logo } from "@/components/layout/Logo";
import { NAV_ITEMS } from "@/components/layout/nav";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { UserMenu } from "@/components/layout/UserMenu";
import { cn } from "@/lib/utils";

/**
 * Responsive frame:
 *  - phone (< md): top bar + bottom tab bar
 *  - tablet (md): narrow icon rail with small labels
 *  - desktop (lg+): full sidebar
 */
export function AppShell() {
  const location = useLocation();
  const section = `/${location.pathname.split("/")[1]}`; // animate between pages, not within one

  return (
    <div className="min-h-dvh bg-background">
      <a
        href="#main"
        className="sr-only z-50 rounded-md bg-primary text-sm font-medium text-primary-foreground focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:px-4 focus:py-2 focus:shadow-lg"
      >
        Skip to content
      </a>

      {/* Sidebar: rail on tablet, full on desktop */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-20 flex-col border-r border-sidebar-border bg-sidebar md:flex lg:w-64">
        <div className="flex h-16 items-center justify-center px-4 lg:justify-start">
          <Logo className="lg:hidden" compact />
          <Logo className="hidden lg:inline-flex" />
        </div>
        <nav aria-label="Main" className="flex-1 px-2 py-4 lg:px-3">
          <ul className="space-y-1">
            {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={end}
                  className={({ isActive }) =>
                    cn(
                      "group flex flex-col items-center gap-1 rounded-lg px-2 py-2.5 text-[11px] font-medium transition-colors outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 lg:flex-row lg:gap-3 lg:px-3 lg:text-sm",
                      isActive
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground",
                    )
                  }
                >
                  <Icon className="size-5 shrink-0" aria-hidden="true" />
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <div className="space-y-3 border-t border-sidebar-border p-3">
          <div className="flex items-center justify-center px-1 lg:justify-between">
            <HealthIndicator compact />
            <span className="hidden lg:block">
              <ThemeToggle />
            </span>
          </div>
          <div className="hidden lg:block">
            <UserMenu showEmail />
          </div>
          <div className="flex flex-col items-center gap-2 lg:hidden">
            <ThemeToggle />
            <UserMenu />
          </div>
        </div>
      </aside>

      {/* Phone top bar */}
      <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b bg-background/85 px-4 backdrop-blur md:hidden">
        <Logo compact />
        <div className="flex items-center gap-1">
          <HealthIndicator compact />
          <ThemeToggle />
          <UserMenu />
        </div>
      </header>

      <div className="md:pl-20 lg:pl-64">
        <main id="main" tabIndex={-1} className="mx-auto w-full max-w-6xl px-4 pt-6 pb-28 outline-none sm:px-6 md:pt-10 md:pb-12 lg:px-10">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={section}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
          <Disclaimer className="mt-12 border-t pt-6" />
        </main>
      </div>

      {/* Phone bottom tab bar */}
      <nav
        aria-label="Main"
        className="fixed inset-x-0 bottom-0 z-30 border-t bg-background/95 pb-[env(safe-area-inset-bottom)] backdrop-blur md:hidden"
      >
        <ul className="grid grid-cols-4">
          {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                className={({ isActive }) =>
                  cn(
                    "flex min-h-14 flex-col items-center justify-center gap-1 text-[11px] font-medium outline-none focus-visible:bg-accent",
                    isActive ? "text-primary" : "text-muted-foreground",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <span
                      className={cn(
                        "flex h-7 w-12 items-center justify-center rounded-full transition-colors",
                        isActive && "bg-accent",
                      )}
                    >
                      <Icon className="size-5" aria-hidden="true" />
                    </span>
                    {label}
                  </>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}
