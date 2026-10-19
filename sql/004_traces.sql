-- 004_traces.sql
-- Step-level traces (one row per span) linked to predictions by trace_id.
-- Run in the Supabase SQL Editor before deploying the app version that sends
-- predictions.trace_id: until then every prediction-log insert fails (the app
-- shows "Could not log prediction") and trace rows are written to traces/ only.

create table if not exists public.traces (
    id              bigint generated always as identity primary key,
    trace_id        uuid not null,
    span_id         text not null,
    parent_span_id  text,
    user_id         uuid not null default auth.uid() references auth.users (id) on delete cascade,
    name            text not null,
    started_at      timestamptz not null,
    latency_ms      real check (latency_ms >= 0),
    status          text not null check (status in ('ok', 'error')),
    error_type      text,
    error           text,
    attributes      jsonb not null default '{}'::jsonb,
    created_at      timestamptz not null default now()
);

create index if not exists traces_trace_id_idx on public.traces (trace_id);
create index if not exists traces_user_started_idx on public.traces (user_id, started_at desc);

comment on table public.traces is
    'One row per pipeline span (preprocess, classify, calibrate, gradcam, report, llm_call, tool_call, guard, fallback_used, ...). attributes holds step details such as token counts and nominal cost.';

-- Row Level Security: each authenticated user can only insert/read their own spans
alter table public.traces enable row level security;

drop policy if exists "Users can insert their own traces" on public.traces;
create policy "Users can insert their own traces"
    on public.traces
    for insert
    to authenticated
    with check (user_id = (select auth.uid()));

drop policy if exists "Users can view their own traces" on public.traces;
create policy "Users can view their own traces"
    on public.traces
    for select
    to authenticated
    using (user_id = (select auth.uid()));

grant select, insert on public.traces to authenticated;

-- Link each logged prediction to its trace (no FK: spans may be written before or after the prediction row)
alter table public.predictions
    add column if not exists trace_id uuid;

create index if not exists predictions_trace_id_idx on public.predictions (trace_id);
