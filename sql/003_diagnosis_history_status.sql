-- 003_diagnosis_history_status.sql
-- Triage status on the per-patient diagnosis history. Run in the Supabase SQL
-- Editor before deploying the app version that sends it: until then every
-- history insert fails (single tab shows an error, batch rows show "Error").

alter table public.diagnosis_history
    add column if not exists status text check (status in ('confident', 'refer'));

comment on column public.diagnosis_history.status is
    'Triage from calibrated confidence: confident (>= 0.9) or refer (clinician review). NULL for records saved before triage existed.';
