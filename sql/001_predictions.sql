-- 001_predictions.sql
-- Prediction logging + daily summary view. Run in the Supabase SQL Editor.

create table if not exists public.predictions (
    id              bigint generated always as identity primary key,
    user_id         uuid not null default auth.uid() references auth.users (id) on delete cascade,
    predicted_label text not null,
    confidence      real not null check (confidence >= 0 and confidence <= 1),
    inference_ms    integer not null check (inference_ms >= 0),
    created_at      timestamptz not null default now()
);

create index if not exists predictions_user_created_idx
    on public.predictions (user_id, created_at);

-- Row Level Security: each authenticated user can only insert/read their own rows
alter table public.predictions enable row level security;

drop policy if exists "Users can insert their own predictions" on public.predictions;
create policy "Users can insert their own predictions"
    on public.predictions
    for insert
    to authenticated
    with check (user_id = (select auth.uid()));

drop policy if exists "Users can view their own predictions" on public.predictions;
create policy "Users can view their own predictions"
    on public.predictions
    for select
    to authenticated
    using (user_id = (select auth.uid()));

grant select, insert on public.predictions to authenticated;

-- Daily summary per label. security_invoker makes the view run with the
-- caller's permissions, so the RLS policies above still apply.
create or replace view public.prediction_daily_summary
with (security_invoker = true) as
select
    (created_at at time zone 'UTC')::date                 as day,
    predicted_label,
    count(*)                                              as prediction_count,
    round(avg(confidence)::numeric, 3)                    as avg_confidence,
    count(*) filter (where confidence < 0.6)              as low_confidence_count,
    round(avg(inference_ms), 1)                           as avg_inference_ms
from public.predictions
group by 1, 2;

grant select on public.prediction_daily_summary to authenticated;
