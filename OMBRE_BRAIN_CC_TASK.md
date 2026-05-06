# 给 Claude Code 沙盒的任务说明：Ombre Brain 改 Supabase 存储

你现在在 Claude 应用里的 Claude Code 沙盒环境，不在真实服务器上。

不要假设你能部署、重启服务、操作 Coolify、读取线上环境变量、访问真实 Supabase secret，或修改线上文件。你的任务是修改用户提供/克隆的 Ombre-Brain fork 仓库代码，并输出清晰的 patch 与说明。

目标仓库：

```text
https://github.com/yuanzi325/Ombre-Brain.git
```

目标：把 Ombre Brain 的主存储从本地 Markdown 文件改成 Supabase `public.memories`，并加双 profile 读取隔离。第一目标是可靠跑通，不要为了最短路径牺牲可运行性，也不要一次性重写所有机制导致难以定位错误。

## 总体原则

- 不要丢现有 `public.memories` 里的 55 条数据。
- 不要把 `public.memories.id` 改成 Ombre 的 12 位 bucket id。
- Ombre 内部继续使用 `bucket_id`，Supabase 表里新增 `bucket_id text unique` 做映射。
- 不要把 `SUPABASE_TABLE` 改成 `memories`。如果代码里需要表名，用 `MEMORY_TABLE=memories`。
- 不要用 `memories.author` 做 cc/cx 隔离，因为 author 是内容作者语义。
- 双 profile 用 `profiles text[]` 做读取视角，例如 `["shared"]`、`["cc"]`、`["cx"]`。
- 第一阶段允许关闭 embedding，先跑通 Supabase CRUD、MCP tools、dehydration。
- 不改前端站点，不改现有 memory-vault MCP，不改 Telegram bridge。
- 不要输出任何 token、secret、env 实际值。

## 第一阶段目标

先把测试服务跑通：

1. Supabase 能创建、读取、更新、软删除、列表、搜索 Ombre bucket。
2. MCP 6 个工具继续可用：`breath` / `hold` / `grow` / `trace` / `pulse` / `dream`。
3. `profile=cc` 只能读 `shared + cc`。
4. `profile=cx` 只能读 `shared + cx`。
5. 默认不传 profile 时只读/写 `shared`。
6. embedding 第一阶段可关闭，不能阻塞主流程。

## 数据库迁移 SQL

给用户一份 SQL，让她在 Supabase 上执行。不要在代码里硬编码 secret。

```sql
alter table public.memories
  add column if not exists bucket_id text,
  add column if not exists bucket_type text default 'dynamic',
  add column if not exists name text,
  add column if not exists domain text[] default '{}',
  add column if not exists tags text[] default '{}',
  add column if not exists valence double precision default 0.5,
  add column if not exists arousal double precision default 0.3,
  add column if not exists activation_count double precision default 0,
  add column if not exists last_active timestamptz,
  add column if not exists resolved boolean default false,
  add column if not exists pinned boolean default false,
  add column if not exists protected boolean default false,
  add column if not exists digested boolean default false,
  add column if not exists model_valence double precision,
  add column if not exists decay_score double precision;

update public.memories
set bucket_id = coalesce(bucket_id, id::text)
where bucket_id is null;

create unique index if not exists memories_bucket_id_key
on public.memories(bucket_id);

create index if not exists memories_bucket_type_idx
on public.memories(bucket_type);

create index if not exists memories_last_active_idx
on public.memories(last_active);

create index if not exists memories_profiles_gin_idx
on public.memories using gin(profiles);

create index if not exists memories_domain_gin_idx
on public.memories using gin(domain);

create index if not exists memories_tags_gin_idx
on public.memories using gin(tags);
```

如果 `public.memories` 已有 `raw jsonb`，继续用 `raw` 存 Ombre 额外 metadata，例如：

```text
raw.ombre_metadata
raw.ombre_deleted
```

## 文件 1：requirements.txt

新增依赖：

```text
supabase>=2.0.0
```

## 文件 2：utils.py

在 `load_config()` 增加这些 env/config：

```text
OMBRE_STORAGE_BACKEND=file|supabase，默认 file
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY 或 SUPABASE_KEY
MEMORY_TABLE，默认 memories
OMBRE_EMBEDDING_ENABLED=true|false，第一阶段建议 false
```

建议输出到 config：

```python
config["storage"]["backend"]
config["supabase"]["url"]
config["supabase"]["key"]
config["supabase"]["memory_table"]
config["embedding"]["enabled"]
```

如果 `OMBRE_STORAGE_BACKEND=supabase`，不要强依赖本地 bucket 文件目录存在；但可以保留 `buckets_dir`，因为 `dehydrator.py` 的 SQLite cache 第一阶段仍可能用它。

## 文件 3：bucket_manager.py

这是核心文件。保持 `BucketManager` 对外 API 尽量不变，让 `server.py` 少改。

### `__init__`

新增：

```python
self.storage_backend = config.get("storage", {}).get("backend", "file")
self.use_supabase = self.storage_backend == "supabase"
```

如果 `use_supabase`：

```python
from supabase import create_client
self.supabase = create_client(url, key)
self.memory_table = config.get("supabase", {}).get("memory_table", "memories")
```

### 新增 `_row_to_bucket(row)`

把 Supabase row 转成 Ombre 原来的 bucket 结构：

```python
{
  "id": row["bucket_id"],
  "metadata": {
    "id": row["bucket_id"],
    "name": row["name"] or row["title"] or row["bucket_id"],
    "tags": row["tags"] or row["keywords"] or [],
    "domain": row["domain"] or [],
    "valence": row["valence"] or 0.5,
    "arousal": row["arousal"] or 0.3,
    "importance": row["importance"] or 5,
    "type": row["bucket_type"] or "dynamic",
    "created": row["created_at"] or row["date"],
    "last_active": row["last_active"] or row["updated_at"] or row["date"],
    "activation_count": row["activation_count"] or 0,
    "resolved": row["resolved"],
    "pinned": row["pinned"],
    "protected": row["protected"],
    "digested": row["digested"],
    "model_valence": row["model_valence"],
    "profiles": row["profiles"] or ["shared"],
  },
  "content": row["content"] or "",
  "path": "",
  "row": row,
}
```

注意空值兜底，避免 `None` 把评分逻辑炸掉。

### 新增 `_metadata_to_row(metadata, content)`

把 Ombre metadata 转成 `public.memories` row：

```text
bucket_id
name
title
content
tags
keywords
domain
bucket_type
importance
valence
arousal
activation_count
last_active
resolved
pinned
protected
digested
model_valence
profiles
raw.ombre_metadata
updated_at
```

不要覆盖现有 `id`。

### 改 `create(..., profiles=None)`

file backend 保持旧逻辑。

supabase backend：

- 生成 `bucket_id = generate_bucket_id()`
- 仍按原逻辑构造 metadata
- `metadata["profiles"] = profiles or ["shared"]`
- insert 到 `public.memories`
- 返回 bucket_id

### 改 `get(bucket_id)`

supabase backend：

- 优先 `.eq("bucket_id", bucket_id)`
- 如果没找到且 bucket_id 看起来像 UUID，可尝试 `.eq("id", bucket_id)`
- 返回 `_row_to_bucket(row)` 或 None

### 改 `update(bucket_id, **kwargs)`

supabase backend：

- 先 `get(bucket_id)`
- 修改 metadata/content
- 支持字段：

```text
content
tags
importance
domain
valence
arousal
name
resolved
pinned
protected
digested
model_valence
profiles
activation_count
```

- pinned/protected 时 importance 锁 10
- 更新 `last_active`
- update `public.memories` where `bucket_id = bucket_id`

### 改 `delete(bucket_id)`

supabase backend 第一阶段不要 hard delete。

做软删除：

```text
raw.ombre_deleted = true
resolved = true
bucket_type = "archived"
updated_at = now
```

如果调用 `embedding_engine.delete_embedding(bucket_id)` 失败，不要阻塞删除。

### 改 `touch(bucket_id)`

supabase backend：

- `activation_count += 1`
- `last_active = now`
- update row
- 调 `_time_ripple`

### 改 `_time_ripple(...)`

supabase backend 不读文件。

- 用 `list_all(include_archive=False)`
- 找 created/last_active 在 ±48h 的 bucket
- 最多 5 条
- `activation_count += 0.3`
- 不递归 touch

### 改 `list_all(include_archive=False)`

supabase backend：

- select `*` from `MEMORY_TABLE`
- 排除 `raw.ombre_deleted == true`
- `include_archive=false` 时排除 `bucket_type == "archived"`
- Python 侧转 `_row_to_bucket`
- 按 `last_active` / `updated_at` / `created_at` 近到远排序

### 改 `archive(bucket_id)`

supabase backend：

```text
bucket_type = "archived"
metadata.type = "archived"
```

### 改 `get_stats()`

supabase backend：

- 用 `list_all(include_archive=True)`
- Python 侧统计 type/domain
- size 用 content 字符数估算即可

### 改 `search(...)`

尽量保留现有搜索/打分逻辑，因为它本来就是 `list_all()` 后 Python 打分。

新增参数：

```python
profile: str = ""
```

在候选 bucket 上做 profile filter。

## 文件 4：profile helper

可以放在 `bucket_manager.py`。

```python
def _visible_for_profile(bucket, profile: str = "") -> bool:
    profiles = bucket.get("metadata", {}).get("profiles") or ["shared"]
    if not profile:
        return "shared" in profiles
    return "shared" in profiles or profile in profiles

def filter_by_profile(buckets, profile: str = ""):
    return [b for b in buckets if _visible_for_profile(b, profile)]
```

默认不传 profile 时只看 shared。  
`profile="cc"` 时看 shared + cc。  
`profile="cx"` 时看 shared + cx。

## 文件 5：server.py

不要大改工具结构，重点是给 6 个工具加 profile 视角。

### 改 `_merge_or_create`

增加参数：

```python
profile: str = ""
profiles: list[str] | None = None
```

要求：

- 搜索候选时 `bucket_mgr.search(..., profile=profile)`
- create 时传 `profiles=profiles or ["shared"]`
- merge 只合并当前 profile 可见的 bucket，避免 cc 私有记忆合并进 cx 私有记忆。

### 改 `breath`

新增参数：

```python
profile: str = ""
```

所有路径都要过滤 profile：

- `importance_min` 模式
- 空 query 自动浮现
- `domain="feel"` 检索
- 普通 search
- vector search 拿到 bucket 后也要检查 profile 可见性
- random surfacing 也要从当前 profile 可见 bucket 里取

### 改 `hold`

新增参数：

```python
profile: str = ""
profiles: str = ""
```

解析逻辑：

```python
def parse_profiles(profiles: str = "", profile: str = "", feel: bool = False) -> list[str]:
    if profiles and profiles.strip():
        return [p.strip() for p in profiles.split(",") if p.strip()]
    if feel and profile and profile.strip():
        return [profile.strip()]
    return ["shared"]
```

规则：

- 普通 hold 默认写入 `["shared"]`
- `feel=True` 且传了 `profile`，默认写入 `[profile]`
- pinned 也要传 profiles
- `_merge_or_create` 也要传 profile/profiles

### 改 `grow`

签名改成：

```python
async def grow(content: str, profile: str = "", profiles: str = "") -> str:
```

短内容 fast path 和 digest 拆分后的每个 item 都传 profile/profiles。

默认 shared。

### 改 `trace`

新增参数：

```python
profiles: str = ""
```

如果传了 profiles，就：

```python
await bucket_mgr.update(bucket_id, profiles=[...])
```

其他字段保持原逻辑。

### 改 `pulse`

签名：

```python
async def pulse(include_archive: bool = False, profile: str = "") -> str:
```

`list_all()` 后按 profile 过滤。

### 改 `dream`

签名：

```python
async def dream(profile: str = "") -> str:
```

只从 shared + 当前 profile 的 bucket 里做梦境/浮现。

### hooks/API

如果有：

- `/breath-hook`
- `/dream-hook`
- `/api/buckets`
- `/api/search`
- `/api/network`

加可选 query 参数 `profile`，然后过滤。

dashboard 的删除逻辑不要直接 `os.remove(file_path)`，统一改成：

```python
await bucket_mgr.delete(bucket_id)
```

`/api/host-vault` 这类本地文件功能，在 Supabase backend 下返回说明即可：

```text
storage backend is supabase; host vault is disabled
```

## 文件 6：embedding_engine.py

第一阶段建议关闭 embedding，不要迁移到 Supabase。

原因：GLM-4-Flash 是 chat model，不保证能直接做 embedding。先跑通主流程。

要求：

- 支持 `OMBRE_EMBEDDING_ENABLED=false`
- disabled 时：
  - `generate_and_store()` 返回 False
  - `search_similar()` 返回 []
  - 不得影响 MCP 工具启动和使用

SQLite embedding db 可以暂时保留。

## 文件 7：dehydrator.py

第一阶段 dehydration cache 可以继续用 SQLite，因为它只是派生缓存，不是主数据。

GLM OpenAI-compatible env：

```text
OMBRE_API_KEY=<智谱 key>
OMBRE_DEHYDRATION_BASE_URL=https://open.bigmodel.cn/api/paas/v4
OMBRE_DEHYDRATION_MODEL=glm-4-flash
```

不要硬编码 key。

## Coolify 测试服务 env

部署时建议先作为测试服务，不要立刻替换现有 memory-vault MCP。

```text
OMBRE_STORAGE_BACKEND=supabase
SUPABASE_URL=<existing>
SUPABASE_SERVICE_ROLE_KEY=<existing>
MEMORY_TABLE=memories
OMBRE_TRANSPORT=streamable-http
OMBRE_PORT=8000
OMBRE_API_KEY=<zhipu key>
OMBRE_DEHYDRATION_BASE_URL=https://open.bigmodel.cn/api/paas/v4
OMBRE_DEHYDRATION_MODEL=glm-4-flash
OMBRE_EMBEDDING_ENABLED=false
OMBRE_DASHBOARD_PASSWORD=<set one>
```

## 验证

本地语法：

```bash
python -m compileall .
```

文件 backend 旧测试：

```bash
OMBRE_STORAGE_BACKEND=file pytest -q
```

Supabase smoke test：

1. 服务启动成功。
2. `GET /health` 成功。
3. MCP `pulse()` 不报错。
4. `hold(content="测试 shared 记忆")` 成功，Supabase 出现 `bucket_id`。
5. `breath(query="测试")` 能查到。
6. `hold(content="cc 私有测试", profile="cc", profiles="cc")`。
7. `hold(content="cx 私有测试", profile="cx", profiles="cx")`。
8. `breath(query="私有测试", profile="cc")` 只能看到 shared + cc。
9. `breath(query="私有测试", profile="cx")` 只能看到 shared + cx。
10. `dream(profile="cc")` 不浮现纯 cx 记忆。
11. `trace(bucket_id=..., resolved=True)` 能更新。
12. `delete(bucket_id)` 是软删除，不 hard delete。

## 交付说明

改完请输出：

- 改了哪些文件。
- 每个文件改了哪些函数。
- Supabase migration SQL。
- 需要的环境变量。
- 第一阶段刻意关闭/暂缓了什么，例如 embedding。
- 你实际跑了哪些验证命令，结果是什么。
- 如果有没跑的测试，说明原因。

