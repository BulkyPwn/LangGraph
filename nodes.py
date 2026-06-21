"""主图节点函数。

演示特性：
- understand：注入 BaseStore（节点签名声明 store 自动注入）读取长期记忆
- plan：LLM 绑定工具（bind_tools），可触发 ToolNode + tools_condition 路由
- approval：interrupt() 暂停等待人工，返回 Command(goto=..., update=...) 动态路由
- revise：Command(goto=...) 反思循环回到 plan
- finalize：生成最终行程
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt

from config import BUDGET_LIMIT, MAX_ITERATIONS, make_llm
from state import TravelState
from store import load_user_preferences
from tools import ALL_TOOLS

DEMO_USER = "demo_user"


def _extract_list(text: str, fallback: list[str]) -> list[str]:
    """从 LLM 文本中尽量解析出 JSON 列表，失败则回退。"""
    try:
        # 取第一个 JSON 数组片段
        m = re.search(r"\[.*?\]", text, re.S)
        if m:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [str(x) for x in data]
    except Exception:
        pass
    return fallback


# ---------------------------------------------------------------------------
# 节点：理解需求
# ---------------------------------------------------------------------------
def understand(state: TravelState, store: BaseStore) -> dict:
    """解析用户诉求，提取目的地；并从长期记忆读取用户偏好。

    节点签名含 `store` 参数 → LangGraph 自动注入 compile 时传入的 BaseStore。
    """
    query = state["query"]
    # —— 长期记忆：跨 thread 读取用户偏好 ——
    prefs = load_user_preferences(store, DEMO_USER) if store is not None else {}

    llm = make_llm(temperature=0.0)
    resp = llm.invoke(
        f"你是需求解析器。请从用户需求中提取旅行目的地，仅输出 JSON 字符串数组，"
        f"例如 [\"北京\",\"上海\"]。用户需求：{query}"
    )
    destinations = _extract_list(resp.content, ["北京"])

    print(f"[understand] 目的地={destinations} 偏好={prefs}")
    return {
        "destinations": destinations,
        "user_preferences": prefs,
        "budget_limit": state.get("budget_limit", BUDGET_LIMIT),
        "iterations": 0,
        "messages": [
            SystemMessage(content=f"已解析目的地：{destinations}；用户偏好：{prefs}")
        ],
    }


# ---------------------------------------------------------------------------
# 节点：汇总并行研究结果，估算预算
# ---------------------------------------------------------------------------
def aggregate(state: TravelState) -> dict:
    """把 Map-Reduce 汇聚到 research_results 的多条结果整合，并给出粗略预算。"""
    results = state.get("research_results", [])
    lines = [f"- {r['destination']}：{r['summary']}" for r in results]
    summary = "\n".join(lines) if lines else "（暂无研究结果）"

    # 粗略预算：每个目的地按 3000 元估算（仅演示）
    budget_total = 3000.0 * len(results) if results else 0.0

    print(f"[aggregate] 共 {len(results)} 个目的地，估算预算 {budget_total}")
    return {
        "budget_total": budget_total,
        "messages": [HumanMessage(content=f"研究汇总：\n{summary}")],
    }


# ---------------------------------------------------------------------------
# 节点：生成行程计划（绑定工具，可被 tools_condition 路由到 ToolNode）
# ---------------------------------------------------------------------------
async def plan(state: TravelState) -> dict:
    """LLM 生成行程计划；若需精确预算可调用 budget_calculator 工具。

    注：本节点为 async 以支持 compile 时的 timeout（同步节点无法安全取消）。
    """
    llm = make_llm().bind_tools(ALL_TOOLS)
    results = state.get("research_results", [])
    prefs = state.get("user_preferences", {})
    research_text = "\n".join(
        f"- {r['destination']}：{r['summary']}" for r in results
    )

    system = SystemMessage(
        content=(
            "你是资深旅行规划师。基于研究结果制定行程计划。"
            "如需精确核算总花费，请调用 budget_calculator 工具（传入各项花费列表）。"
        )
    )
    user = HumanMessage(
        content=(
            f"用户偏好：{prefs}\n"
            f"预算上限：{state.get('budget_limit', BUDGET_LIMIT)}\n"
            f"研究汇总：\n{research_text}\n"
            f"请给出一份行程计划。"
        )
    )
    resp = await llm.ainvoke([system, user])
    print(f"[plan] 生成计划，是否含工具调用：{bool(getattr(resp, 'tool_calls', None))}")
    return {"messages": [resp]}


# ---------------------------------------------------------------------------
# 节点：人工审批（interrupt + Command 动态路由）
# ---------------------------------------------------------------------------
def approval(state: TravelState) -> Command:
    """暂停图执行，等待人工决策；依据结果用 Command 动态跳转。

    演示特性：
    - interrupt()：把上下文抛给调用方，等待 Command(resume=...) 续跑
    - Command(goto=...)：节点内部控制下一个去哪个节点（动态路由）
    """
    decision = interrupt(
        {
            "prompt": "行程计划已生成，请审批",
            "budget_total": state.get("budget_total", 0.0),
            "budget_limit": state.get("budget_limit", BUDGET_LIMIT),
        }
    )
    approved = str(decision).strip().lower() in ("yes", "y", "true", "1", "批准", "同意", "ok")
    goto = "finalize" if approved else "revise"
    print(f"[approval] 人工决策={decision!r} → approved={approved} → goto={goto}")
    return Command(
        update={"approved": approved, "messages": [HumanMessage(content=f"人工审批结果：{approved}")]},
        goto=goto,
    )


# ---------------------------------------------------------------------------
# 节点：反思修订（Command 回到 plan，构成循环）
# ---------------------------------------------------------------------------
def revise(state: TravelState) -> Command:
    """审批未通过时回到 plan 重新规划；超过上限次数则强制收尾。"""
    it = state.get("iterations", 0) + 1
    if it >= MAX_ITERATIONS:
        print(f"[revise] 达到最大迭代 {MAX_ITERATIONS}，强制收尾")
        return Command(goto="finalize", update={"iterations": it})
    print(f"[revise] 第 {it} 次修订，回到 plan")
    return Command(goto="plan", update={"iterations": it, "approved": None})


# ---------------------------------------------------------------------------
# 节点：生成最终行程文案
# ---------------------------------------------------------------------------
def finalize(state: TravelState) -> dict:
    """汇总全流程，产出最终行程文案。"""
    llm = make_llm()
    approved = state.get("approved")
    results = state.get("research_results", [])
    research_text = "\n".join(f"- {r['destination']}：{r['summary']}" for r in results)

    resp = llm.invoke(
        [
            SystemMessage(content="你是旅行规划师，请把最终确定的行程整理成一段清晰文案。"),
            HumanMessage(
                content=(
                    f"审批状态：{approved}\n"
                    f"预算：{state.get('budget_total')}（上限 {state.get('budget_limit')}）\n"
                    f"研究：\n{research_text}\n"
                    f"请输出最终行程。"
                )
            ),
        ]
    )
    print("[finalize] 已生成最终行程")
    return {"final_itinerary": resp.content, "messages": [resp]}
