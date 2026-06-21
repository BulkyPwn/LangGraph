"""主图组装。

演示特性（本文件集中体现）：
- StateGraph + input_schema≠output_schema（TravelInput / TravelOutput）
- Send：Map-Reduce 并行扇出到 research 子图
- 条件边：route_fanout（扇出）、route_after_plan（工具路由）、route_approval（审批路由）
- 子图作为节点：research（StateGraph 子图）、evaluate_budget（Functional 子图）
- ToolNode + 自定义 tools_condition 路由
- RetryPolicy：给 tools 节点配置重试（应对 budget_calculator 模拟失败）
- timeout：给 plan 节点配置超时
- Checkpointer：SqliteSaver（SQLite 持久化）
- BaseStore：长期记忆注入
- interrupt_before：编译期断点（可选，演示用）
- Command：由 approval / revise 节点在节点内部完成动态跳转
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy, Send

from config import SQLITE_PATH, BUDGET_LIMIT
from nodes import aggregate, approval, finalize, plan, revise, understand
from state import TravelInput, TravelOutput, TravelState
from store import GLOBAL_STORE
from subgraphs import build_budget_subgraph, build_research_subgraph
from tools import ALL_TOOLS


# ---------------------------------------------------------------------------
# 路由函数
# ---------------------------------------------------------------------------
def route_fanout(state: TravelState):
    """从 understand 出发：把每个目的地用 Send 并行发给 research 子图。

    无目的地时退化为直接进入 aggregate。
    返回 list[Send] 即触发并行 Map-Reduce。
    """
    dests = state.get("destinations", []) or []
    if not dests:
        return "aggregate"
    return [Send("research", {"destinations": [d]}) for d in dests]


def route_after_plan(state: TravelState) -> str:
    """plan 之后：若 LLM 发起工具调用则去 tools 节点，否则进入预算评估。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "evaluate_budget"


def route_approval(state: TravelState) -> str:
    """预算评估之后：需要审批则进入 approval（含 interrupt），否则直接收尾。"""
    return "approval" if state.get("needs_approval") else "finalize"


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------
def build_graph(
    *,
    checkpointer=None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
):
    """构建并编译主图。

    参数：
        checkpointer:   检查点保存器（SqliteSaver / InMemorySaver）
        store:          长期记忆 BaseStore
        interrupt_before: 编译期断点（演示 interrupt_before 特性）
    """
    g = StateGraph(TravelState, input_schema=TravelInput, output_schema=TravelOutput)

    # —— 节点 ——
    g.add_node("understand", understand)
    # 子图作为节点
    g.add_node("research", build_research_subgraph())
    g.add_node("aggregate", aggregate)
    # plan 节点：配置 timeout（秒），演示节点超时保护
    g.add_node("plan", plan, timeout=60)
    # tools 节点：配置 RetryPolicy，应对 budget_calculator 模拟失败
    g.add_node("tools", ToolNode(ALL_TOOLS), retry_policy=RetryPolicy(max_attempts=5, backoff_factor=1.0))
    # 预算评估子图（Functional API）
    g.add_node("evaluate_budget", build_budget_subgraph())
    g.add_node("approval", approval)
    g.add_node("revise", revise)
    g.add_node("finalize", finalize)

    # —— 边 ——
    g.add_edge(START, "understand")
    # understand → 并行扇出（Send）或 aggregate
    g.add_conditional_edges("understand", route_fanout)
    # 并行研究完成后汇聚
    g.add_edge("research", "aggregate")
    g.add_edge("aggregate", "plan")
    # plan → 工具 / 预算评估
    g.add_conditional_edges("plan", route_after_plan)
    # 工具执行后回到 plan（ReAct 风格循环）
    g.add_edge("tools", "plan")
    # 预算评估 → 审批 / 收尾
    g.add_conditional_edges("evaluate_budget", route_approval)
    # approval / revise 通过 Command(goto=...) 自行决定去向，无需显式出边
    g.add_edge("finalize", END)

    return g.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
    )


def make_sqlite_checkpointer(path: str = SQLITE_PATH) -> SqliteSaver:
    """创建并初始化【同步】SQLite 检查点保存器（仅用于同步执行场景）。"""
    conn = sqlite3.connect(path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()  # 建表
    return checkpointer


async def make_async_sqlite_checkpointer(path: str = SQLITE_PATH):
    """创建并初始化【异步】SQLite 检查点保存器。

    异步执行（astream/ainvoke）必需：SqliteSaver 不支持 async 方法，
    而 timeout 等特性要求异步执行，因此主流程使用 AsyncSqliteSaver。
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = await aiosqlite.connect(path)
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()  # 建表
    return checkpointer

