"""长期记忆封装：基于 BaseStore（InMemoryStore）实现跨 thread 持久记忆。

演示特性：
- BaseStore（短/长期记忆分离）：与 Checkpointer 不同，Store 跨 thread 共享
- put / get / search：写入、读取、按命名空间检索用户偏好

Checkpointer 保存的是「单次会话的执行状态」；
Store 保存的是「跨会话的长期事实」，例如用户始终偏好“省钱/自由行”。
"""
from __future__ import annotations

import json

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# 全局单例 store（跨 thread 共享）
GLOBAL_STORE: BaseStore = InMemoryStore()

# 命名空间约定：("user", <user_id>, "preferences")
USER_PREF_NS = ("user", "preferences")


def save_user_preference(store: BaseStore, user_id: str, key: str, value) -> None:
    """把单个偏好字段写入长期记忆。"""
    ns = USER_PREF_NS + (user_id,)
    existing = store.get(ns, "profile")
    profile: dict = json.loads(existing.value["data"]) if existing else {}
    profile[key] = value
    store.put(ns, "profile", {"data": json.dumps(profile, ensure_ascii=False)})


def load_user_preferences(store: BaseStore, user_id: str) -> dict:
    """读取某用户全部偏好。"""
    ns = USER_PREF_NS + (user_id,)
    item = store.get(ns, "profile")
    if not item:
        return {}
    return json.loads(item.value["data"])


def search_users(store: BaseStore, keyword: str) -> list[str]:
    """按关键词检索（演示 store.search）。"""
    hits = store.search(USER_PREF_NS, query=keyword)
    return [item.key for item in hits]
