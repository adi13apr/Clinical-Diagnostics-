# Web UI (React)

The browser front end for the FastAPI backend in `../api`. Plain JavaScript (JSX), no TypeScript.

**Stack:** Vite · React 19 · Tailwind CSS 4 · shadcn/ui (JSX, `tsx: false`) · Framer Motion · Recharts · TanStack Query · React Router · supabase-js · ESLint (with `jsx-a11y`).

## Run it

```bash
# 1. Start the API (from the repo root)
pip install -r requirements-api.txt
uvicorn api.main:create_app --factory --port 8000

# 2. Start the web app
cd web
cp .env.example .env.local      # then fill in the three values
npm install
npm run dev                     # http://localhost:5173
```

| Variable | Meaning |
| --- | --- |
| `VITE_API_URL` | Where the API runs, e.g. `http://localhost:8000` |
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon key (public by design; row-level security protects the data) |

Other scripts: `npm run lint` (ESLint), `npm run build` (production build in `dist/`), `npm run preview` (serve that build).

**Google sign-in** works in the browser with PKCE. In Supabase → Authentication → URL Configuration → Redirect URLs, add `http://localhost:5173/**` and the deployed site's URL with `/**`. The API must also allow the deployed origin: set `FRONTEND_URL` on the API (localhost is always allowed).

## Pages

| Route | What it does |
| --- | --- |
| `/login` | Landing hero, sign in / create account, Google sign-in |
| `/` Diagnose | Drag-and-drop upload → result card with confidence gauge, Confident/Refer banner, per-class probabilities; Grad-CAM overlay with an opacity slider; AI report; PDF download |
| `/history` | Your predictions: filter by class, triage and date (server side), search the loaded page, paginate. Filters live in the URL |
| `/analytics` | Stat tiles, predictions per day (stacked by class), referral rate, time per pipeline step, report pipeline numbers. Every chart has a table view |
| `/traces/:id` | Step timeline (waterfall) per request with status and latency; click a step for its details |

## How it works

- **Auth:** `supabase-js` keeps the session; `src/lib/api.js` sends its access token as `Authorization: Bearer …` on every call except `/health`.
- **Server state:** TanStack Query hooks in `src/hooks/queries.js`. Query keys include the user id, and the cache is cleared on sign-out.
- **Diagnose state** lives in `DiagnosisProvider`, above the router, so a result (or a report that takes a minute) survives a visit to another page.
- **Grad-CAM overlay:** the page fetches the raw 7×7 map (`GET /gradcam/{id}/map`) and draws it in the browser over the image you uploaded, with the same bilinear resize and jet colormap as the Python code (`src/lib/colormap.js`). That's what makes the opacity slider possible.
- **PDF:** built in the browser with jsPDF (loaded only when you click). The optional patient name is never sent to the server; it only goes into the file.
- **Probabilities** are never shown as 100% or 0% (`probability()` in `src/lib/format.js`), and they round the same way as the report text, so the page and the report agree.

## Design

- Calm blue/teal palette, defined as CSS variables in `src/index.css`, with a separate dark palette. Light/dark/system is chosen in the header and remembered in this browser.
- Class colours (Normal / Pneumonia / Tuberculosis) were checked with a colour-blindness validator in both themes. Two are below 3:1 contrast in light mode, so charts always have a legend and a table view, and text never uses the series colour.
- Triage status always has an icon and a word, never colour alone.
- Motion is subtle (fades, slides, the gauge filling in, skeleton loaders) and turns off for people who set "reduce motion".
- Layout: bottom tab bar on phones, an icon rail on tablets, a full sidebar on desktop.
- Keyboard: a skip link, visible focus rings, a dropzone that is a real button (Enter opens the file picker), and arrow keys on the opacity slider.

## Layout

```
src/
  main.jsx            providers: Theme → Query → Auth → Diagnosis → Router
  router.jsx          routes; History, Analytics and Traces are lazy-loaded
  index.css           Tailwind + theme tokens (light and dark)
  lib/                api.js (JSDoc client), supabase.js, format.js, colormap.js, pdf.js, utils.js
  providers/          ThemeProvider, AuthProvider, DiagnosisProvider, contexts.js (hooks)
  hooks/              queries.js (TanStack Query), useCssVars.js (theme colours for charts)
  components/
    ui/               shadcn/ui components (generated, JSX)
    layout/           AppShell (sidebar / rail / bottom nav), ThemeToggle, UserMenu, RequireAuth
    common/           PageHeader, EmptyState/ErrorState, StatusBadge/ClassLabel, Disclaimer
    diagnose/         UploadDropzone, ResultCard, ConfidenceGauge, TriageBanner, ProbabilityBars,
                      GradcamViewer, ReportPanel
    history/          HistoryFilters, HistoryTable
    analytics/        ChartCard, Charts, StatTiles
    trace/            traceModel.js, TraceTimeline
  pages/              LoginPage, DiagnosePage, HistoryPage, AnalyticsPage, TracePage, NotFoundPage
```

Add more shadcn components with `npx shadcn@latest add <name>`. `components.json` sets `tsx: false`, so they arrive as JSX. Check the `cn` import afterwards: on this setup the CLI wrote `from "cn"` instead of `from "@/lib/utils"`.
