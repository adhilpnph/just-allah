create extension if not exists vector;

-- One table for everything. kind: 'quran' | 'hadith' | 'athar' (sahaba reports)
create table passages (
  id         bigserial primary key,
  kind       text not null,
  ref        text not null unique,   -- 'quran:2:255', 'bukhari:1', 'athar:muwatta:12'
  collection text,                   -- 'Sahih al-Bukhari'
  arabic     text not null,
  english    text not null,
  narrator   text,
  grade      text,                   -- 'sahih' | 'hasan' | 'daif' | 'mawdu'; null for Quran
  graded_by  text,                   -- 'Albani', 'Bukhari', ...
  grades     jsonb,
  url        text,
  embedding  vector(384),            -- multilingual-e5-small
  fts        tsvector generated always as (to_tsvector('english', english)) stored
);

create index on passages using hnsw (embedding vector_cosine_ops);
create index on passages using gin (fts);
create index on passages (kind);

-- Hybrid search (vector + keyword, reciprocal rank fusion) within one kind.
-- The app calls it 3 times in priority order: quran, hadith, athar.
create function search_passages(q text, qe vector(384), p_kind text, k int default 5)
returns table (ref text, kind text, collection text, arabic text, english text,
               narrator text, grade text, graded_by text, grades jsonb, url text)
language sql stable as $$
  with v as (
    select id, row_number() over (order by embedding <=> qe) as r
    from passages
    where kind = p_kind
    order by embedding <=> qe
    limit 20
  ),
  t as (
    select id, row_number() over (order by ts_rank(fts, websearch_to_tsquery('english', q)) desc) as r
    from passages
    where kind = p_kind and fts @@ websearch_to_tsquery('english', q)
    limit 20
  ),
  scored as (
    select id, sum(1.0 / (60 + r)) as score
    from (select * from v union all select * from t) x
    group by id
  )
  select p.ref, p.kind, p.collection, p.arabic, p.english,
         p.narrator, p.grade, p.graded_by, p.grades, p.url
  from scored s
  join passages p using (id)
  order by s.score desc
  limit k;
$$;

-- Citation check: the LLM returns only refs; the app renders text from the DB.
-- Any ref that returns no row is dropped.
-- select ref, arabic, english, grade, graded_by from passages where ref = any($1);