"""主图状态定义 + 自定义 Reducer。

演示特性：
- TypedDict 状态
- Annotated 自定义 reducer：
    * add_messages          —— 消息累加（LangGraph 内置）
    * operator.add          —— 列表/数值合并（用于 Map-Reduce 并行结果汇聚）
    * merge_plan (自定义)    —— 自定义合并逻辑（新计划覆盖旧计划，但保留历史）
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


def merge_plan(left: dict | None, right: dict | None) -> dict:
    """自定义 reducer：新计划覆盖旧计划字段，并把旧计划追加到 history。

    这是一个演示性的自定义合并函数：当节点返回新计划时，旧计划被压入
    plan_history，从而演示“自定义 reducer 不止能 add/overwrite”。
    """
    if right is None:
        return left or {}
    if left is None:
        left = {}
    merged = {**left, **right}
    # 把被覆盖的旧计划留痕
    if left and "title" in left and left.get("title") != right.get("title"):
        merged.setdefault("plan_history", []).append(left.get("title"))
    return merged


class TravelState(TypedDict):
    """旅行规划助手的主图状态。"""
    # —— 消息历史（使用内置 add_messages reducer）——
    messages: Annotated[list, add_messages]

    # —— 用户原始诉求 ——
    query: str

    # —— 规划阶段：拆分出的目的地（并行研究的输入）——
    destinations: list[str]

    # —— Map-Reduce 并行研究结果（operator.add 汇聚各 Send 任务输出）——
    research_results: Annotated[list[dict], operator.add]

    # —— 行程计划（自定义 merge_plan reducer）——
    plan: Annotated[dict, merge_plan]

    # —— 预算 ——
    budget_total: float
    budget_limit: float

    # —— 人工审批 ——
    needs_approval: bool
    approved: Annotated[bool | None, lambda l, r: r if r is not None else l]

    # —— 反思循环计数 ——
    iterations: int

    # —— 从长期记忆读出的用户偏好 ——
    user_preferences: dict

    # —— 最终行程文案 ——
    final_itinerary: str


# 输入 schema（更轻量）：外部只需提供 query + 预算上限
class TravelInput(TypedDict):
    query: str
    budget_limit: float


# 输出 schema（富结构）：对外暴露最终结果
class TravelOutput(TypedDict):
    final_itinerary: str
    plan: dict
    budget_total: float
    approved: bool | None
