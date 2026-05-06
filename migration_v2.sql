-- ============================================================
-- Ombre Brain Migration v2
-- 目标表: public.memories (Supabase)
-- 安全：全部 IF NOT EXISTS，可重复执行
-- ============================================================

-- Step 1: Add all Ombre columns (safe to re-run)
alter table public.memories
  add column if not exists profiles         text[]            default '{"shared"}',
  add column if not exists importance       integer           default 5,
  add column if not exists bucket_id        text,
  add column if not exists bucket_type      text              default 'dynamic',
  add column if not exists name             text,
  add column if not exists domain           text[]            default '{}',
  add column if not exists tags             text[]            default '{}',
  add column if not exists valence          double precision  default 0.5,
  add column if not exists arousal          double precision  default 0.3,
  add column if not exists activation_count double precision  default 0,
  add column if not exists last_active      timestamptz,
  add column if not exists resolved         boolean           default false,
  add column if not exists pinned           boolean           default false,
  add column if not exists protected        boolean           default false,
  add column if not exists digested         boolean           default false,
  add column if not exists model_valence    double precision,
  add column if not exists decay_score      double precision;

-- Step 2: Backfill bucket_id from id for existing rows without one
update public.memories
set bucket_id = id::text
where bucket_id is null;

-- Step 3: Backfill profiles for existing rows (default to shared)
update public.memories
set profiles = '{"shared"}'
where profiles is null or profiles = '{}';

-- Step 4: Indexes

-- Unique bucket_id lookup (primary Ombre key)
create unique index if not exists memories_bucket_id_key
  on public.memories(bucket_id);

-- bucket_type filter (list_all excludes archived)
create index if not exists memories_bucket_type_idx
  on public.memories(bucket_type);

-- time-based queries (touch, ripple, list)
create index if not exists memories_last_active_idx
  on public.memories(last_active);

-- importance filter (breath importance_min mode)
create index if not exists memories_importance_idx
  on public.memories(importance);

-- GIN indexes for array columns

-- profile isolation filter (most critical)
create index if not exists memories_profiles_gin_idx
  on public.memories using gin(profiles);

-- domain filter pre-pass in search
create index if not exists memories_domain_gin_idx
  on public.memories using gin(domain);

-- tags search
create index if not exists memories_tags_gin_idx
  on public.memories using gin(tags);
