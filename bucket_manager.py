# ============================================================
# Module: Memory Bucket Manager (bucket_manager.py)
# 模块：记忆桶管理器
#
# CRUD operations, multi-dimensional index search, activation updates
# for memory buckets. Supports file backend (default) and Supabase backend.
#
# Depended on by: server.py, decay_engine.py
# ============================================================

import os
import math
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import frontmatter
from rapidfuzz import fuzz

from utils import generate_bucket_id, sanitize_name, safe_path, now_iso

logger = logging.getLogger("ombre_brain.bucket")


class BucketManager:
    def __init__(self, config: dict, embedding_engine=None):
        # --- Storage paths (file backend) ---
        self.base_dir = config["buckets_dir"]
        self.permanent_dir = os.path.join(self.base_dir, "permanent")
        self.dynamic_dir = os.path.join(self.base_dir, "dynamic")
        self.archive_dir = os.path.join(self.base_dir, "archive")
        self.feel_dir = os.path.join(self.base_dir, "feel")
        self.fuzzy_threshold = config.get("matching", {}).get("fuzzy_threshold", 50)
        self.max_results = config.get("matching", {}).get("max_results", 5)

        # --- Supabase backend ---
        self.storage_backend = config.get("storage", {}).get("backend", "file")
        self.use_supabase = self.storage_backend == "supabase"

        if self.use_supabase:
            from supabase import create_client
            url = config["supabase"]["url"]
            key = config["supabase"]["key"]
            self.supabase = create_client(url, key)
            self.memory_table = config.get("supabase", {}).get("memory_table", "memories")
            logger.info(f"BucketManager: Supabase backend, table={self.memory_table}")

        # --- Wikilink config ---
        wikilink_cfg = config.get("wikilink", {})
        self.wikilink_enabled = wikilink_cfg.get("enabled", True)
        self.wikilink_use_tags = wikilink_cfg.get("use_tags", False)
        self.wikilink_use_domain = wikilink_cfg.get("use_domain", True)
        self.wikilink_use_auto_keywords = wikilink_cfg.get("use_auto_keywords", True)
        self.wikilink_auto_top_k = wikilink_cfg.get("auto_top_k", 8)
        self.wikilink_min_len = wikilink_cfg.get("min_keyword_len", 2)
        self.wikilink_exclude_keywords = set(wikilink_cfg.get("exclude_keywords", []))
        self.wikilink_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "我们", "你们", "他们", "然后", "今天", "昨天", "明天", "一下",
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "with",
            "this", "that", "from", "they", "been", "said", "will", "each",
        }
        self.wikilink_stopwords |= {w.lower() for w in self.wikilink_exclude_keywords}

        # --- Search scoring weights ---
        scoring = config.get("scoring_weights", {})
        self.w_topic = scoring.get("topic_relevance", 4.0)
        self.w_emotion = scoring.get("emotion_resonance", 2.0)
        self.w_time = scoring.get("time_proximity", 1.5)
        self.w_importance = scoring.get("importance", 1.0)
        self.content_weight = scoring.get("content_weight", 1.0)

        self.embedding_engine = embedding_engine

    # =========================================================
    # Profile helpers
    # =========================================================

    @staticmethod
    def _visible_for_profile(bucket: dict, profile: str = "") -> bool:
        """
        No profile → only "shared" buckets visible.
        profile="cc" → "shared" + "cc" visible.
        profile="cx" → "shared" + "cx" visible.
        """
        profiles = bucket.get("metadata", {}).get("profiles") or ["shared"]
        if not profile:
            return "shared" in profiles
        return "shared" in profiles or profile in profiles

    @staticmethod
    def filter_by_profile(buckets: list, profile: str = "") -> list:
        return [b for b in buckets if BucketManager._visible_for_profile(b, profile)]

    # =========================================================
    # Supabase row ↔ bucket conversion
    # =========================================================

    def _row_to_bucket(self, row: dict) -> dict:
        """Convert a public.memories row to Ombre bucket structure."""
        raw = row.get("raw") or {}
        ombre_meta = raw.get("ombre_metadata") or {}

        bucket_id = row.get("bucket_id") or str(row.get("id", ""))

        # importance: column first, then ombre_metadata, default 5
        importance = row.get("importance")
        if importance is None:
            importance = ombre_meta.get("importance", 5)
        try:
            importance = max(1, min(10, int(importance)))
        except (TypeError, ValueError):
            importance = 5

        created = str(row.get("created_at") or row.get("date") or "")
        last_active = (
            str(row.get("last_active") or "")
            or str(row.get("updated_at") or "")
            or created
        )

        activation_count = row.get("activation_count")
        if activation_count is None:
            activation_count = ombre_meta.get("activation_count", 0)
        try:
            activation_count = float(activation_count)
        except (TypeError, ValueError):
            activation_count = 0.0

        return {
            "id": bucket_id,
            "metadata": {
                "id": bucket_id,
                "name": row.get("name") or row.get("title") or bucket_id,
                "tags": row.get("tags") or row.get("keywords") or [],
                "domain": row.get("domain") or [],
                "valence": float(row.get("valence") if row.get("valence") is not None else 0.5),
                "arousal": float(row.get("arousal") if row.get("arousal") is not None else 0.3),
                "importance": importance,
                "type": row.get("bucket_type") or "dynamic",
                "created": created,
                "last_active": last_active,
                "activation_count": activation_count,
                "resolved": bool(row.get("resolved")),
                "pinned": bool(row.get("pinned")),
                "protected": bool(row.get("protected")),
                "digested": bool(row.get("digested")),
                "model_valence": row.get("model_valence"),
                "profiles": row.get("profiles") or ["shared"],
            },
            "content": row.get("content") or "",
            "path": "",
            "row": row,
        }

    def _metadata_to_row(self, metadata: dict, content: str, existing_raw: dict = None) -> dict:
        """Convert Ombre metadata + content to a public.memories row dict.
        Only includes columns guaranteed by migration SQL + standard columns.
        importance/keywords stored in raw.ombre_metadata (not as direct columns).
        """
        raw = dict(existing_raw or {})
        # Store full metadata snapshot + extra fields not in schema columns
        raw["ombre_metadata"] = {
            **metadata,
            "importance": metadata.get("importance", 5),
        }

        return {
            "bucket_id": metadata.get("id") or metadata.get("bucket_id"),
            "name": metadata.get("name"),
            "title": metadata.get("name"),
            "content": content,
            "tags": metadata.get("tags") or [],
            "domain": metadata.get("domain") or [],
            "bucket_type": metadata.get("type") or "dynamic",
            "importance": max(1, min(10, int(metadata.get("importance", 5)))),
            "valence": max(0.0, min(1.0, float(metadata.get("valence", 0.5)))),
            "arousal": max(0.0, min(1.0, float(metadata.get("arousal", 0.3)))),
            "activation_count": metadata.get("activation_count", 0),
            "last_active": metadata.get("last_active") or now_iso(),
            "resolved": bool(metadata.get("resolved", False)),
            "pinned": bool(metadata.get("pinned", False)),
            "protected": bool(metadata.get("protected", False)),
            "digested": bool(metadata.get("digested", False)),
            "model_valence": metadata.get("model_valence"),
            "profiles": metadata.get("profiles") or ["shared"],
            "raw": raw,
            "updated_at": now_iso(),
        }

    # =========================================================
    # Create
    # =========================================================

    async def create(
        self,
        content: str,
        tags: list[str] = None,
        importance: int = 5,
        domain: list[str] = None,
        valence: float = 0.5,
        arousal: float = 0.3,
        bucket_type: str = "dynamic",
        name: str = None,
        pinned: bool = False,
        protected: bool = False,
        profiles: list[str] = None,
    ) -> str:
        """Create a new memory bucket, return bucket ID."""
        bucket_id = generate_bucket_id()
        bucket_name = sanitize_name(name) if name else bucket_id

        if bucket_type == "feel":
            domain = domain if domain is not None else []
        else:
            domain = domain or ["未分类"]
        tags = tags or []

        if pinned or protected:
            importance = 10

        metadata = {
            "id": bucket_id,
            "name": bucket_name,
            "tags": tags,
            "domain": domain,
            "valence": max(0.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "importance": max(1, min(10, importance)),
            "type": bucket_type,
            "created": now_iso(),
            "last_active": now_iso(),
            "activation_count": 0,
            "profiles": profiles or ["shared"],
        }
        if pinned:
            metadata["pinned"] = True
            if bucket_type != "permanent":
                metadata["type"] = "permanent"
        if protected:
            metadata["protected"] = True

        if self.use_supabase:
            row = self._metadata_to_row(metadata, content)
            row["created_at"] = metadata["created"]
            try:
                self.supabase.table(self.memory_table).insert(row).execute()
            except Exception as e:
                logger.error(f"Supabase insert failed for {bucket_id}: {e}")
                raise
            logger.info(f"Created bucket (supabase): {bucket_id} ({bucket_name})")
            return bucket_id

        # --- File backend ---
        post = frontmatter.Post(content, **metadata)

        if bucket_type == "permanent" or pinned:
            type_dir = self.permanent_dir
        elif bucket_type == "feel":
            type_dir = self.feel_dir
        else:
            type_dir = self.dynamic_dir

        if bucket_type == "feel":
            primary_domain = "沉淀物"
        else:
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"

        target_dir = os.path.join(type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)

        if bucket_name and bucket_name != bucket_id:
            filename = f"{bucket_name}_{bucket_id}.md"
        else:
            filename = f"{bucket_id}.md"
        file_path = safe_path(target_dir, filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket file: {file_path}: {e}")
            raise

        logger.info(
            f"Created bucket: {bucket_id} ({bucket_name}) → {primary_domain}/"
            + (" [PINNED]" if pinned else "") + (" [PROTECTED]" if protected else "")
        )
        return bucket_id

    # =========================================================
    # Get
    # =========================================================

    async def get(self, bucket_id: str) -> Optional[dict]:
        """Read a single bucket by ID."""
        if not bucket_id or not isinstance(bucket_id, str):
            return None

        if self.use_supabase:
            try:
                result = self.supabase.table(self.memory_table).select("*").eq("bucket_id", bucket_id).execute()
                if result.data:
                    row = result.data[0]
                    if (row.get("raw") or {}).get("ombre_deleted"):
                        return None
                    return self._row_to_bucket(row)
                # Fallback: bucket_id might be a UUID
                if len(bucket_id) in (32, 36):
                    result2 = self.supabase.table(self.memory_table).select("*").filter("id", "eq", bucket_id).execute()
                    if result2.data:
                        row = result2.data[0]
                        if (row.get("raw") or {}).get("ombre_deleted"):
                            return None
                        return self._row_to_bucket(row)
                return None
            except Exception as e:
                logger.warning(f"Supabase get failed for {bucket_id}: {e}")
                return None

        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return None
        return self._load_bucket(file_path)

    # =========================================================
    # Update
    # =========================================================

    async def update(self, bucket_id: str, **kwargs) -> bool:
        """Update bucket content or metadata fields."""
        if self.use_supabase:
            return await self._update_supabase(bucket_id, **kwargs)

        # --- File backend ---
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            post = frontmatter.load(file_path)
        except Exception as e:
            logger.warning(f"Failed to load bucket for update: {file_path}: {e}")
            return False

        is_pinned = post.get("pinned", False) or post.get("protected", False)
        if is_pinned:
            kwargs.pop("importance", None)

        if "content" in kwargs:
            post.content = kwargs["content"]
        if "tags" in kwargs:
            post["tags"] = kwargs["tags"]
        if "importance" in kwargs:
            post["importance"] = max(1, min(10, int(kwargs["importance"])))
        if "domain" in kwargs:
            post["domain"] = kwargs["domain"]
        if "valence" in kwargs:
            post["valence"] = max(0.0, min(1.0, float(kwargs["valence"])))
        if "arousal" in kwargs:
            post["arousal"] = max(0.0, min(1.0, float(kwargs["arousal"])))
        if "name" in kwargs:
            post["name"] = sanitize_name(kwargs["name"])
        if "resolved" in kwargs:
            post["resolved"] = bool(kwargs["resolved"])
        if "pinned" in kwargs:
            post["pinned"] = bool(kwargs["pinned"])
            if kwargs["pinned"]:
                post["importance"] = 10
        if "digested" in kwargs:
            post["digested"] = bool(kwargs["digested"])
        if "model_valence" in kwargs:
            post["model_valence"] = max(0.0, min(1.0, float(kwargs["model_valence"])))
        if "profiles" in kwargs:
            post["profiles"] = kwargs["profiles"]

        post["last_active"] = now_iso()

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket update: {file_path}: {e}")
            return False

        domain = post.get("domain", ["未分类"])
        if kwargs.get("pinned") and post.get("type") != "permanent":
            post["type"] = "permanent"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
            self._move_bucket(file_path, self.permanent_dir, domain)

        logger.info(f"Updated bucket: {bucket_id}")
        return True

    async def _update_supabase(self, bucket_id: str, **kwargs) -> bool:
        try:
            result = self.supabase.table(self.memory_table).select("*").eq("bucket_id", bucket_id).execute()
            if not result.data:
                return False
            row = result.data[0]
        except Exception as e:
            logger.warning(f"Supabase fetch for update failed {bucket_id}: {e}")
            return False

        is_pinned = bool(row.get("pinned")) or bool(row.get("protected"))
        updates: dict = {}

        if "content" in kwargs:
            updates["content"] = kwargs["content"]
        if "tags" in kwargs:
            updates["tags"] = kwargs["tags"]
        if "domain" in kwargs:
            updates["domain"] = kwargs["domain"]
        if "name" in kwargs:
            updates["name"] = sanitize_name(kwargs["name"])
            updates["title"] = updates["name"]
        if "resolved" in kwargs:
            updates["resolved"] = bool(kwargs["resolved"])
        if "pinned" in kwargs:
            updates["pinned"] = bool(kwargs["pinned"])
            if kwargs["pinned"]:
                updates["importance"] = 10  # pinned → lock importance column to 10
                updates["bucket_type"] = "permanent"
        if "protected" in kwargs:
            updates["protected"] = bool(kwargs["protected"])
            if kwargs["protected"]:
                updates["importance"] = 10  # protected → lock importance column to 10
        if "digested" in kwargs:
            updates["digested"] = bool(kwargs["digested"])
        if "model_valence" in kwargs:
            updates["model_valence"] = max(0.0, min(1.0, float(kwargs["model_valence"])))
        if "profiles" in kwargs:
            updates["profiles"] = kwargs["profiles"]
        if "activation_count" in kwargs:
            updates["activation_count"] = kwargs["activation_count"]
        if "valence" in kwargs:
            updates["valence"] = max(0.0, min(1.0, float(kwargs["valence"])))
        if "arousal" in kwargs:
            updates["arousal"] = max(0.0, min(1.0, float(kwargs["arousal"])))
        if "importance" in kwargs and not is_pinned:
            updates["importance"] = max(1, min(10, int(kwargs["importance"])))

        updates["last_active"] = now_iso()
        updates["updated_at"] = now_iso()

        # Update raw.ombre_metadata snapshot (surgical merge, preserve ombre_deleted etc.)
        existing_raw = dict(row.get("raw") or {})
        current_meta = dict(existing_raw.get("ombre_metadata") or {})
        for key in ("name", "tags", "domain", "valence", "arousal", "importance",
                    "resolved", "pinned", "protected", "digested", "model_valence",
                    "profiles", "activation_count", "content"):
            if key in kwargs:
                current_meta[key] = kwargs[key]
        existing_raw["ombre_metadata"] = current_meta
        updates["raw"] = existing_raw

        try:
            self.supabase.table(self.memory_table).update(updates).eq("bucket_id", bucket_id).execute()
        except Exception as e:
            logger.error(f"Supabase update failed for {bucket_id}: {e}")
            return False

        logger.info(f"Updated bucket (supabase): {bucket_id}")
        return True

    # =========================================================
    # Delete (supabase = soft delete)
    # =========================================================

    async def delete(self, bucket_id: str) -> bool:
        """Supabase: soft delete. File: hard delete."""
        if self.use_supabase:
            try:
                result = self.supabase.table(self.memory_table).select("id, raw").eq("bucket_id", bucket_id).execute()
                if not result.data:
                    return False
                existing_raw = dict(result.data[0].get("raw") or {})
                existing_raw["ombre_deleted"] = True
                self.supabase.table(self.memory_table).update({
                    "raw": existing_raw,
                    "resolved": True,
                    "bucket_type": "archived",
                    "updated_at": now_iso(),
                }).eq("bucket_id", bucket_id).execute()
                logger.info(f"Soft-deleted bucket (supabase): {bucket_id}")
                if self.embedding_engine:
                    try:
                        self.embedding_engine.delete_embedding(bucket_id)
                    except Exception:
                        pass
                return True
            except Exception as e:
                logger.error(f"Supabase delete failed for {bucket_id}: {e}")
                return False

        # --- File backend: hard delete ---
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False
        try:
            os.remove(file_path)
        except OSError as e:
            logger.error(f"Failed to delete bucket file: {file_path}: {e}")
            return False
        logger.info(f"Deleted bucket: {bucket_id}")
        return True

    # =========================================================
    # Touch
    # =========================================================

    async def touch(self, bucket_id: str) -> None:
        """Update activation time and count; trigger time ripple."""
        if self.use_supabase:
            try:
                result = self.supabase.table(self.memory_table).select(
                    "activation_count, created_at, last_active"
                ).eq("bucket_id", bucket_id).execute()
                if not result.data:
                    return
                row = result.data[0]
                try:
                    cur_count = float(row.get("activation_count") or 0)
                except (TypeError, ValueError):
                    cur_count = 0.0
                new_now = now_iso()
                self.supabase.table(self.memory_table).update({
                    "activation_count": cur_count + 1,
                    "last_active": new_now,
                    "updated_at": new_now,
                }).eq("bucket_id", bucket_id).execute()
                ref_str = str(row.get("created_at") or row.get("last_active") or "")
                try:
                    ref_time = datetime.fromisoformat(ref_str) if ref_str else datetime.now()
                except (ValueError, TypeError):
                    ref_time = datetime.now()
                await self._time_ripple(bucket_id, ref_time)
            except Exception as e:
                logger.warning(f"Supabase touch failed {bucket_id}: {e}")
            return

        # --- File backend ---
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return
        try:
            post = frontmatter.load(file_path)
            post["last_active"] = now_iso()
            post["activation_count"] = post.get("activation_count", 0) + 1
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
            current_time = datetime.fromisoformat(
                str(post.get("created", post.get("last_active", "")))
            )
            await self._time_ripple(bucket_id, current_time)
        except Exception as e:
            logger.warning(f"Failed to touch bucket: {bucket_id}: {e}")

    async def _time_ripple(self, source_id: str, reference_time: datetime, hours: float = 48.0) -> None:
        """Boost activation_count (+0.3) of time-adjacent buckets. Max 5."""
        try:
            all_buckets = await self.list_all(include_archive=False)
        except Exception:
            return

        rippled = 0
        for bucket in all_buckets:
            if rippled >= 5:
                break
            if bucket["id"] == source_id:
                continue
            meta = bucket.get("metadata", {})
            if meta.get("pinned") or meta.get("protected") or meta.get("type") in ("permanent", "feel"):
                continue

            created_str = meta.get("created", meta.get("last_active", ""))
            try:
                created = datetime.fromisoformat(str(created_str))
                delta_hours = abs((reference_time - created).total_seconds()) / 3600
            except (ValueError, TypeError):
                continue

            if delta_hours > hours:
                continue

            if self.use_supabase:
                try:
                    res = self.supabase.table(self.memory_table).select(
                        "activation_count"
                    ).eq("bucket_id", bucket["id"]).execute()
                    if res.data:
                        cur = float(res.data[0].get("activation_count") or 1)
                        self.supabase.table(self.memory_table).update({
                            "activation_count": round(cur + 0.3, 1)
                        }).eq("bucket_id", bucket["id"]).execute()
                        rippled += 1
                except Exception:
                    continue
            else:
                file_path = self._find_bucket_file(bucket["id"])
                if not file_path:
                    continue
                try:
                    post = frontmatter.load(file_path)
                    cur = post.get("activation_count", 1)
                    post["activation_count"] = round(cur + 0.3, 1)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(frontmatter.dumps(post))
                    rippled += 1
                except Exception:
                    continue

    # =========================================================
    # Search
    # =========================================================

    async def search(
        self,
        query: str,
        limit: int = None,
        domain_filter: list[str] = None,
        query_valence: float = None,
        query_arousal: float = None,
        profile: str = "",
    ) -> list[dict]:
        """
        Multi-dimensional indexed search.
        profile="" → shared only. profile="cc" → shared+cc.
        """
        if not query or not query.strip():
            return []

        limit = limit or self.max_results
        all_buckets = await self.list_all(include_archive=False)

        if not all_buckets:
            return []

        # Profile filter (always applied; default "" = shared only)
        all_buckets = self.filter_by_profile(all_buckets, profile)
        if not all_buckets:
            return []

        # Domain pre-filter
        if domain_filter:
            filter_set = {d.lower() for d in domain_filter}
            candidates = [
                b for b in all_buckets
                if {d.lower() for d in b["metadata"].get("domain", [])} & filter_set
            ]
            if not candidates:
                candidates = all_buckets
        else:
            candidates = all_buckets

        # Embedding pre-filter (optional)
        if self.embedding_engine and self.embedding_engine.enabled:
            try:
                vector_results = await self.embedding_engine.search_similar(query, top_k=50)
                if vector_results:
                    vector_ids = {bid for bid, _ in vector_results}
                    emb_candidates = [b for b in candidates if b["id"] in vector_ids]
                    if emb_candidates:
                        candidates = emb_candidates
            except Exception as e:
                logger.warning(f"Embedding pre-filter failed: {e}")

        # Multi-dim ranking
        scored = []
        for bucket in candidates:
            meta = bucket.get("metadata", {})
            try:
                topic_score = self._calc_topic_score(query, bucket)
                emotion_score = self._calc_emotion_score(query_valence, query_arousal, meta)
                time_score = self._calc_time_score(meta)
                importance_score = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                total = (
                    topic_score * self.w_topic
                    + emotion_score * self.w_emotion
                    + time_score * self.w_time
                    + importance_score * self.w_importance
                )
                weight_sum = self.w_topic + self.w_emotion + self.w_time + self.w_importance
                normalized = (total / weight_sum) * 100 if weight_sum > 0 else 0

                if normalized >= self.fuzzy_threshold:
                    if meta.get("resolved", False):
                        normalized *= 0.3
                    bucket["score"] = round(normalized, 2)
                    scored.append(bucket)
            except Exception as e:
                logger.warning(f"Scoring failed for {bucket.get('id', '?')}: {e}")
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # =========================================================
    # List all
    # =========================================================

    async def list_all(self, include_archive: bool = False) -> list[dict]:
        """List all non-deleted buckets."""
        if self.use_supabase:
            try:
                query = self.supabase.table(self.memory_table).select("*")
                if not include_archive:
                    query = query.neq("bucket_type", "archived")
                result = query.execute()
                rows = result.data or []
                # Exclude soft-deleted rows (Python-side, JSONB filter is complex)
                rows = [r for r in rows if not (r.get("raw") or {}).get("ombre_deleted")]
                buckets = [self._row_to_bucket(r) for r in rows]
                buckets.sort(
                    key=lambda b: (
                        b["metadata"].get("last_active")
                        or b["metadata"].get("created")
                        or ""
                    ),
                    reverse=True,
                )
                return buckets
            except Exception as e:
                logger.error(f"Supabase list_all failed: {e}")
                return []

        # --- File backend ---
        buckets = []
        dirs = [self.permanent_dir, self.dynamic_dir, self.feel_dir]
        if include_archive:
            dirs.append(self.archive_dir)
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    if not filename.endswith(".md"):
                        continue
                    bucket = self._load_bucket(os.path.join(root, filename))
                    if bucket:
                        buckets.append(bucket)
        return buckets

    # =========================================================
    # Stats
    # =========================================================

    async def get_stats(self) -> dict:
        """Return memory bucket statistics."""
        if self.use_supabase:
            try:
                all_buckets = await self.list_all(include_archive=True)
                stats = {
                    "permanent_count": 0,
                    "dynamic_count": 0,
                    "archive_count": 0,
                    "feel_count": 0,
                    "total_size_kb": 0.0,
                    "domains": {},
                }
                for b in all_buckets:
                    btype = b["metadata"].get("type", "dynamic")
                    stats["total_size_kb"] += len(b.get("content") or "") / 1024
                    if btype == "permanent" or b["metadata"].get("pinned"):
                        stats["permanent_count"] += 1
                    elif btype == "feel":
                        stats["feel_count"] += 1
                    elif btype == "archived":
                        stats["archive_count"] += 1
                    else:
                        stats["dynamic_count"] += 1
                    for d in b["metadata"].get("domain", []):
                        stats["domains"][d] = stats["domains"].get(d, 0) + 1
                return stats
            except Exception as e:
                logger.error(f"Supabase get_stats failed: {e}")
                return {
                    "permanent_count": 0, "dynamic_count": 0,
                    "archive_count": 0, "feel_count": 0,
                    "total_size_kb": 0.0, "domains": {},
                }

        # --- File backend ---
        stats = {
            "permanent_count": 0,
            "dynamic_count": 0,
            "archive_count": 0,
            "feel_count": 0,
            "total_size_kb": 0.0,
            "domains": {},
        }
        for subdir, key in [
            (self.permanent_dir, "permanent_count"),
            (self.dynamic_dir, "dynamic_count"),
            (self.archive_dir, "archive_count"),
            (self.feel_dir, "feel_count"),
        ]:
            if not os.path.exists(subdir):
                continue
            for root, _, files in os.walk(subdir):
                for f in files:
                    if f.endswith(".md"):
                        stats[key] += 1
                        try:
                            stats["total_size_kb"] += os.path.getsize(
                                os.path.join(root, f)
                            ) / 1024
                        except OSError:
                            pass
                        domain_name = os.path.basename(root)
                        if domain_name != os.path.basename(subdir):
                            stats["domains"][domain_name] = (
                                stats["domains"].get(domain_name, 0) + 1
                            )
        return stats

    # =========================================================
    # Archive
    # =========================================================

    async def archive(self, bucket_id: str) -> bool:
        """Move bucket to archived state."""
        if self.use_supabase:
            try:
                result = self.supabase.table(self.memory_table).select(
                    "raw"
                ).eq("bucket_id", bucket_id).execute()
                if not result.data:
                    return False
                existing_raw = dict(result.data[0].get("raw") or {})
                meta = dict(existing_raw.get("ombre_metadata") or {})
                meta["type"] = "archived"
                existing_raw["ombre_metadata"] = meta
                self.supabase.table(self.memory_table).update({
                    "bucket_type": "archived",
                    "raw": existing_raw,
                    "updated_at": now_iso(),
                }).eq("bucket_id", bucket_id).execute()
                logger.info(f"Archived bucket (supabase): {bucket_id}")
                return True
            except Exception as e:
                logger.error(f"Supabase archive failed for {bucket_id}: {e}")
                return False

        # --- File backend ---
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False
        try:
            post = frontmatter.load(file_path)
            domain = post.get("domain", ["未分类"])
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"
            archive_subdir = os.path.join(self.archive_dir, primary_domain)
            os.makedirs(archive_subdir, exist_ok=True)
            dest = safe_path(archive_subdir, os.path.basename(file_path))
            post["type"] = "archived"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
            shutil.move(file_path, str(dest))
        except Exception as e:
            logger.error(f"Failed to archive bucket: {bucket_id}: {e}")
            return False
        logger.info(f"Archived bucket: {bucket_id} → archive/{primary_domain}/")
        return True

    # =========================================================
    # File backend internals
    # =========================================================

    def _move_bucket(self, file_path: str, target_type_dir: str, domain: list[str] = None) -> str:
        primary_domain = sanitize_name(domain[0]) if domain else "未分类"
        target_dir = os.path.join(target_type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        new_path = safe_path(target_dir, filename)
        if os.path.normpath(file_path) != os.path.normpath(new_path):
            os.rename(file_path, new_path)
            logger.info(f"Moved bucket: {filename} → {target_dir}/")
        return new_path

    def _find_bucket_file(self, bucket_id: str) -> Optional[str]:
        if not bucket_id:
            return None
        for dir_path in [self.permanent_dir, self.dynamic_dir, self.archive_dir, self.feel_dir]:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if not fname.endswith(".md"):
                        continue
                    name_part = fname[:-3]
                    if name_part == bucket_id or name_part.endswith(f"_{bucket_id}"):
                        return os.path.join(root, fname)
        return None

    def _load_bucket(self, file_path: str) -> Optional[dict]:
        try:
            post = frontmatter.load(file_path)
            return {
                "id": post.get("id", Path(file_path).stem),
                "metadata": dict(post.metadata),
                "content": post.content,
                "path": file_path,
            }
        except Exception as e:
            logger.warning(f"Failed to load bucket file: {file_path}: {e}")
            return None

    # =========================================================
    # Scoring (used by both backends via search)
    # =========================================================

    def _calc_topic_score(self, query: str, bucket: dict) -> float:
        meta = bucket.get("metadata", {})
        name_score = fuzz.partial_ratio(query, meta.get("name", "")) * 3
        domain_score = (
            max((fuzz.partial_ratio(query, d) for d in meta.get("domain", [])), default=0) * 2.5
        )
        tag_score = (
            max((fuzz.partial_ratio(query, tag) for tag in meta.get("tags", [])), default=0) * 2
        )
        content_score = (
            fuzz.partial_ratio(query, bucket.get("content", "")[:1000]) * self.content_weight
        )
        return (name_score + domain_score + tag_score + content_score) / (
            100 * (3 + 2.5 + 2 + self.content_weight)
        )

    def _calc_emotion_score(self, q_valence: float, q_arousal: float, meta: dict) -> float:
        if q_valence is None or q_arousal is None:
            return 0.5
        try:
            b_valence = float(meta.get("valence", 0.5))
            b_arousal = float(meta.get("arousal", 0.3))
        except (ValueError, TypeError):
            return 0.5
        dist = math.sqrt((q_valence - b_valence) ** 2 + (q_arousal - b_arousal) ** 2)
        return max(0.0, 1.0 - dist / 1.414)

    def _calc_time_score(self, meta: dict) -> float:
        last_active_str = meta.get("last_active", meta.get("created", ""))
        try:
            last_active = datetime.fromisoformat(str(last_active_str))
            days = max(0.0, (datetime.now() - last_active).total_seconds() / 86400)
        except (ValueError, TypeError):
            days = 30
        return math.exp(-0.02 * days)
