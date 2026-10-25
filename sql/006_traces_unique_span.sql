-- 006_traces_unique_span.sql
-- One row per (trace_id, span_id), so writing the same spans twice is harmless.
--
-- With a report worker (REDIS_URL set), the worker keeps a report's spans in Redis and the API
-- writes them to Supabase as the user when the browser polls the finished job. Two polls can
-- arrive together, and a failed write is retried on the next poll, so the API inserts with
-- ON CONFLICT (trace_id, span_id) DO NOTHING (supabase-py: upsert(..., ignore_duplicates=True)).
--
-- Run in the Supabase SQL Editor BEFORE deploying the code that sends that upsert: without this
-- index every trace write fails ("there is no unique or exclusion constraint matching the ON
-- CONFLICT specification"). The app and the API only warn, but traces stop reaching Supabase.
--
-- No grant or policy changes: ON CONFLICT DO NOTHING needs only INSERT (plus SELECT for the
-- returned rows), which 004 already grants to authenticated, and the insert policy's WITH CHECK
-- still applies to every inserted row.

-- Drop exact duplicates first (keeps the oldest row), or the unique index can't be built
delete from public.traces a
    using public.traces b
    where a.trace_id = b.trace_id
      and a.span_id = b.span_id
      and a.id > b.id;

create unique index if not exists traces_trace_span_key on public.traces (trace_id, span_id);
