"""研究子图：经典 StateGraph 子图（compiled graph as node）。

演示特性：
- 子图（编译后的图作为节点嵌入父图）
- 父图通过 Send 并行触发本子图（Map-Reduce），每份结果经
  operator.add reducer 汇聚到父状态 research_results
- 子图内部多节点：gather（调工具）→ summarize（LLM 摘要）
"""
from __future__ import annotations

from typing import Annotated, TypedDict

import operator

from langgraph.graph import END, START, StateGraph

from config import make_llm
from tools import get_weather, search_attractions


class ResearchState(TypedDict):
    """研究子图状态：与父图共享 destinations / research_results 两个键。"""
    destinations: list[str]
    # 子图内部暂存
    _scratch: dict
    # 共享键：结果汇聚到父图
    research_results: Annotated[list[dict], operator.add]


class ResearchOutput(TypedDict):
    """子图对外输出：仅 research_results，避免把 destinations（无 reducer）
    回传父图而在并行 Send 汇聚时触发 InvalidUpdateError。"""
    research_results: list[dict]


def gather(state: ResearchState) -> dict:
    """调用工具收集某目的地的天气与景点信息。"""
    dest = state["destinations"][0]
    weather = get_weather.invoke({"city": dest})
    attractions = search_attractions.invoke({"city": dest})
    return {"_scratch": {"destination": dest, "weather": weather, "attractions": attractions}}


def summarize(state: ResearchState) -> dict:
    """用 LLM 把原始信息整理为一段研究摘要。"""
    scratch = state["_scratch"]
    llm = make_llm()
    prompt = (
        f"请为“{scratch['destination']}”撰写一段简短旅行研究摘要，"
        f"结合以下信息：{scratch['weather']}；景点：{scratch['attractions']}。"
        f"不超过 60 字。"
    )
    resp = llm.invoke(prompt)
    return {
        "research_results": [
            {
                "destination": scratch["destination"],
                "weather": scratch["weather"],
                "attractions": scratch["attractions"],
                "summary": resp.content,
            }
        ]
    }


def build_research_subgraph():
    """构建并编译研究子图。"""
    g = StateGraph(ResearchState, output_schema=ResearchOutput)
    g.add_node("gather", gather)
    g.add_node("summarize", summarize)
    g.add_edge(START, "gather")
    g.add_edge("gather", "summarize")
    g.add_edge("summarize", END)
    return g.compile()
