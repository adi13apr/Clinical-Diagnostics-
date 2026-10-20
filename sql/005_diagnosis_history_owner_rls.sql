-- 005_diagnosis_history_owner_rls.sql
-- Each radiologist can only insert and read their own diagnosis_history rows.
-- The old policies ("Public authenticated insert/select") allowed any logged-in
-- user to read every row, and the app only filtered by email client-side. The
-- app used to reach this table with the service_role key, which skips RLS; it now
-- uses the logged-in client (DataService(auth_provider.client)), so these
-- policies are what protects the rows. Safe to run before or after deploying
-- that app change.

alter table public.diagnosis_history enable row level security;

drop policy if exists "Public authenticated insert" on public.diagnosis_history;
drop policy if exists "Public authenticated select" on public.diagnosis_history;

-- Rows are keyed by email (no user_id column); the JWT's email claim is the logged-in user's
drop policy if exists "Radiologists can insert their own history" on public.diagnosis_history;
create policy "Radiologists can insert their own history"
    on public.diagnosis_history
    for insert
    to authenticated
    with check (lower(radiologist_email) = lower((select auth.jwt() ->> 'email')));

drop policy if exists "Radiologists can view their own history" on public.diagnosis_history;
create policy "Radiologists can view their own history"
    on public.diagnosis_history
    for select
    to authenticated
    using (lower(radiologist_email) = lower((select auth.jwt() ->> 'email')));

grant select, insert on public.diagnosis_history to authenticated;
revoke all on public.diagnosis_history from anon;
