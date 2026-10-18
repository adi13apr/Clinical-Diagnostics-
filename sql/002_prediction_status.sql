-- 002_prediction_status.sql
-- Triage status + calibration temperature on the prediction log. Run in the
-- Supabase SQL Editor after 001_predictions.sql, and before deploying the app
-- version that sends these columns (until then every log insert fails and the
-- app shows "Could not log prediction").

alter table public.predictions
    add column if not exists status      text check (status in ('confident', 'refer')),
    add column if not exists temperature real check (temperature > 0);

comment on column public.predictions.status is
    'Triage from calibrated max probability: confident (>= 0.9) or refer. NULL for rows logged before calibration.';
comment on column public.predictions.temperature is
    'Temperature used to calibrate confidence. NULL means confidence is the raw, uncalibrated softmax.';

-- Same view as 001 with refer_count appended (create or replace view can only add columns at the end).
create or replace view public.prediction_daily_summary
with (security_invoker = true) as
select
    (created_at at time zone 'UTC')::date                 as day,
    predicted_label,
    count(*)                                              as prediction_count,
    round(avg(confidence)::numeric, 3)                    as avg_confidence,
    count(*) filter (where confidence < 0.6)              as low_confidence_count,
    round(avg(inference_ms), 1)                           as avg_inference_ms,
    count(*) filter (where status = 'refer')              as refer_count
from public.predictions
group by 1, 2;

grant select on public.prediction_daily_summary to authenticated;
