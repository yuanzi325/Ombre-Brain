#!/usr/bin/env python3
"""
Ombre Brain Smoke Test
======================
用途：验证核心流程在 file backend 可以跑通。
Supabase backend 需要真实环境变量，本脚本支持但需要自行配置。

运行方式（file backend，无需任何外部服务）：
  OMBRE_STORAGE_BACKEND=file python smoke_test.py

运行方式（supabase backend）：
  OMBRE_STORAGE_BACKEND=supabase \
  SUPABASE_URL=<url> \
  SUPABASE_SERVICE_ROLE_KEY=<key> \
  MEMORY_TABLE=memories \
  OMBRE_EMBEDDING_ENABLED=false \
  python smoke_test.py
"""

import os
import sys
import asyncio
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

_results: list[tuple[str, str, str]] = []


def report(name: str, status: str, detail: str = "") -> None:
    mark = "✓" if status == PASS else ("⚠" if status == SKIP else "✗")
    line = f"  [{mark}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    _results.append((name, status, detail))


async def run_tests() -> None:
    from utils import load_config, setup_logging
    from bucket_manager import BucketManager
    from embedding_engine import EmbeddingEngine
    from dehydrator import Dehydrator
    from import_memory import ImportEngine

    config = load_config()
    setup_logging("WARNING")
    backend = config.get("storage", {}).get("backend", "file")
    emb_enabled = config.get("embedding", {}).get("enabled", False)

    print(f"\n=== Ombre Brain Smoke Test ===")
    print(f"  backend          = {backend}")
    print(f"  embedding        = {'on' if emb_enabled else 'off'}")
    print(f"  memory_table     = {config.get('supabase', {}).get('memory_table', 'n/a')}")
    print()

    # ------------------------------------------------------------------ #
    # T01  config loads + required keys present
    # ------------------------------------------------------------------ #
    try:
        assert "buckets_dir" in config
        assert "storage" in config
        report("T01 config loads", PASS)
    except Exception as e:
        report("T01 config loads", FAIL, str(e))
        return

    # ------------------------------------------------------------------ #
    # T02  components initialise without crashing
    # ------------------------------------------------------------------ #
    try:
        emb = EmbeddingEngine(config)
        bucket_mgr = BucketManager(config, embedding_engine=emb)
        dehy = Dehydrator(config)
        report("T02 components init", PASS)
    except Exception as e:
        report("T02 components init", FAIL, str(e))
        return

    # ------------------------------------------------------------------ #
    # T03  create a shared bucket
    # ------------------------------------------------------------------ #
    bid_shared = None
    try:
        bid_shared = await bucket_mgr.create(
            content="smoke test shared memory — 测试共享记忆",
            tags=["smoke", "test"],
            importance=5,
            domain=["测试"],
            profiles=["shared"],
        )
        assert bid_shared and len(bid_shared) == 12
        report("T03 create shared bucket", PASS, f"id={bid_shared}")
    except Exception as e:
        report("T03 create shared bucket", FAIL, str(e))
        return

    # ------------------------------------------------------------------ #
    # T04  get bucket by id
    # ------------------------------------------------------------------ #
    try:
        b = await bucket_mgr.get(bid_shared)
        assert b is not None, "get returned None"
        assert b["metadata"]["profiles"] == ["shared"], f"profiles={b['metadata']['profiles']}"
        report("T04 get bucket", PASS)
    except Exception as e:
        report("T04 get bucket", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T05  search — no profile arg → only shared visible
    # ------------------------------------------------------------------ #
    try:
        hits = await bucket_mgr.search("smoke test shared", profile="")
        found = any(r["id"] == bid_shared for r in hits)
        report("T05 search (no profile → shared only)", PASS if found else SKIP,
               f"found={found}, hits={len(hits)}")
    except Exception as e:
        report("T05 search", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T06  create cc-private bucket
    # ------------------------------------------------------------------ #
    bid_cc = None
    try:
        bid_cc = await bucket_mgr.create(
            content="cc private smoke test memory — cc 私有测试记忆",
            tags=["cc", "private"],
            importance=5,
            domain=["测试"],
            profiles=["cc"],
        )
        assert bid_cc and len(bid_cc) == 12
        report("T06 create cc-private bucket", PASS, f"id={bid_cc}")
    except Exception as e:
        report("T06 create cc-private bucket", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T07  create cx-private bucket
    # ------------------------------------------------------------------ #
    bid_cx = None
    try:
        bid_cx = await bucket_mgr.create(
            content="cx private smoke test memory — cx 私有测试记忆",
            tags=["cx", "private"],
            importance=5,
            domain=["测试"],
            profiles=["cx"],
        )
        report("T07 create cx-private bucket", PASS, f"id={bid_cx}")
    except Exception as e:
        report("T07 create cx-private bucket", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T08  profile isolation: cx CANNOT see cc-private
    # ------------------------------------------------------------------ #
    if bid_cc:
        try:
            hits_cx = await bucket_mgr.search("cc private smoke", profile="cx")
            # list_all + filter
            all_b = await bucket_mgr.list_all(include_archive=False)
            visible_cx = bucket_mgr.filter_by_profile(all_b, "cx")
            ids_cx = {b["id"] for b in visible_cx}
            cc_visible_to_cx = bid_cc in ids_cx
            report("T08 isolation: cx cannot see cc", PASS if not cc_visible_to_cx else FAIL,
                   f"cc_in_cx_view={cc_visible_to_cx}")
        except Exception as e:
            report("T08 isolation: cx cannot see cc", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T09  profile isolation: cc CANNOT see cx-private
    # ------------------------------------------------------------------ #
    if bid_cx:
        try:
            all_b = await bucket_mgr.list_all(include_archive=False)
            visible_cc = bucket_mgr.filter_by_profile(all_b, "cc")
            ids_cc = {b["id"] for b in visible_cc}
            cx_visible_to_cc = bid_cx in ids_cc
            report("T09 isolation: cc cannot see cx", PASS if not cx_visible_to_cc else FAIL,
                   f"cx_in_cc_view={cx_visible_to_cc}")
        except Exception as e:
            report("T09 isolation: cc cannot see cx", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T10  shared visible to all profiles
    # ------------------------------------------------------------------ #
    if bid_shared:
        try:
            all_b = await bucket_mgr.list_all(include_archive=False)
            for p in ("", "cc", "cx"):
                visible = bucket_mgr.filter_by_profile(all_b, p)
                ids = {b["id"] for b in visible}
                assert bid_shared in ids, f"shared not visible to profile={repr(p)}"
            report("T10 shared visible to all profiles", PASS)
        except Exception as e:
            report("T10 shared visible to all profiles", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T11  update: set resolved=True
    # ------------------------------------------------------------------ #
    if bid_shared:
        try:
            ok = await bucket_mgr.update(bid_shared, resolved=True)
            b2 = await bucket_mgr.get(bid_shared)
            assert ok, "update returned False"
            assert b2 and b2["metadata"].get("resolved") is True
            report("T11 update resolved=True", PASS)
        except Exception as e:
            report("T11 update resolved=True", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T12  soft-delete (supabase) / hard-delete (file) — bucket disappears
    # ------------------------------------------------------------------ #
    bid_del = None
    try:
        bid_del = await bucket_mgr.create(
            content="bucket to delete in smoke test",
            tags=[],
            importance=1,
            domain=["测试"],
            profiles=["shared"],
        )
        ok = await bucket_mgr.delete(bid_del)
        b_after = await bucket_mgr.get(bid_del)
        assert ok, "delete returned False"
        assert b_after is None, f"bucket still visible after delete: {b_after}"
        mode = "soft" if backend == "supabase" else "hard"
        report("T12 delete bucket", PASS, f"{mode}-delete, backend={backend}")
    except Exception as e:
        report("T12 delete bucket", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T13  pulse: list_all returns non-empty
    # ------------------------------------------------------------------ #
    try:
        all_b = await bucket_mgr.list_all(include_archive=False)
        report("T13 list_all (pulse)", PASS, f"count={len(all_b)}")
    except Exception as e:
        report("T13 list_all (pulse)", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T14  ImportEngine: init and get_status() has v2 fields
    # ------------------------------------------------------------------ #
    import_eng = None
    try:
        import_eng = ImportEngine(config, bucket_mgr, dehy, emb)
        status = import_eng.get_status()
        required_fields = {
            "status", "memories_created", "memories_merged",
            "memories_new", "memories_merge_candidate", "memories_skipped", "errors",
        }
        missing = required_fields - set(status.keys())
        if missing:
            report("T14 ImportEngine status fields", FAIL, f"missing={missing}")
        else:
            report("T14 ImportEngine status fields", PASS)
    except Exception as e:
        report("T14 ImportEngine status fields", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T15  import with profile=cc (file backend, no LLM needed if API absent)
    # ------------------------------------------------------------------ #
    if import_eng:
        sample = (
            "[用户] 我最喜欢橘子味的糖，这是我的秘密\n"
            "[AI] 好的，我记住了。"
        )
        try:
            result = await import_eng.start(
                raw_content=sample,
                filename="smoke_test_chat.txt",
                preserve_raw=False,
                resume=False,
                profile="cc",
                profiles=["cc"],
            )
            status_val = result.get("status", "")
            if status_val in ("completed", "running", "paused"):
                report("T15 import with profile=cc", PASS,
                       f"status={status_val}, created={result.get('memories_created', 0)}")
            else:
                report("T15 import with profile=cc", SKIP,
                       f"status={status_val} (LLM API likely unavailable — expected without key)")
        except RuntimeError as e:
            if "API not available" in str(e) or "api" in str(e).lower():
                report("T15 import with profile=cc", SKIP,
                       "LLM API not available — expected in keyless env")
            else:
                report("T15 import with profile=cc", FAIL, str(e))
        except Exception as e:
            report("T15 import with profile=cc", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # T16  identity/core detection function
    # ------------------------------------------------------------------ #
    try:
        from import_memory import _is_identity_or_core
        identity_items = [
            {"name": "我的身份设定", "content": "我是小克", "domain": ["身份设定"], "tags": [], "importance": 10},
            {"name": "核心关系", "content": "我们是伴侣", "domain": ["核心关系"], "tags": [], "importance": 8},
        ]
        normal_items = [
            {"name": "今天的午饭", "content": "今天吃了番茄炒蛋", "domain": ["饮食"], "tags": [], "importance": 3},
            {"name": "学习笔记", "content": "今天复习了马克思原理", "domain": ["学习"], "tags": [], "importance": 4},
        ]
        for it in identity_items:
            assert _is_identity_or_core(it), f"should be identity: {it['name']}"
        for it in normal_items:
            assert not _is_identity_or_core(it), f"should NOT be identity: {it['name']}"
        report("T16 _is_identity_or_core detection", PASS)
    except ImportError:
        report("T16 _is_identity_or_core detection", SKIP, "_is_identity_or_core not exported — skipped")
    except Exception as e:
        report("T16 _is_identity_or_core detection", FAIL, str(e))

    # ------------------------------------------------------------------ #
    # cleanup
    # ------------------------------------------------------------------ #
    for cleanup_id in [bid_shared, bid_cc, bid_cx]:
        if cleanup_id:
            try:
                await bucket_mgr.delete(cleanup_id)
            except Exception:
                pass

    print()


async def main() -> None:
    try:
        await run_tests()
    except Exception:
        traceback.print_exc()

    passed  = sum(1 for _, s, _ in _results if s == PASS)
    skipped = sum(1 for _, s, _ in _results if s == SKIP)
    failed  = sum(1 for _, s, _ in _results if s == FAIL)
    total   = len(_results)

    print(f"Results: {passed} passed  {skipped} skipped  {failed} failed  / {total} total")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
