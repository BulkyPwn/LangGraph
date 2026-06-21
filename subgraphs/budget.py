"""预算子图：Functional API（@entrypoint + @task）。

演示特性：
- LangGraph 第二套 API：Functional API（函数式编排）
- @entrypoint 定义图入口，@task 定义可持久化的任务单元
- 作为子图节点嵌入父图（compiled functional graph as node）
- 与父图共享键：budget_total / budget_limit / needs_approval
"""
from __future__ import annotations

from langgraph.func import entrypoint, task

from config import BUDGET_LIMIT


@task
def decide_approval(budget_total: float, budget_limit: float) -> bool:
    """@task：判断是否需要人工审批（超出预算上限即需要）。

    被包裹为 @task 后，在 @entrypoint 中调用可获得持久化/重放能力。
    """
    over_budget = budget_total > budget_limit
    print(f"   [budget @task] 预算 {budget_total} vs 上限 {budget_limit} → 需审批={over_budget}")
    return over_budget


@entrypoint()
def evaluate_budget(state: dict) -> dict:
    """@entrypoint：预算评估子图入口。

    输入为父图状态（共享键自动流入），返回对共享键的更新。
    """
    budget_total = state.get("budget_total", 0.0)
    budget_limit = state.get("budget_limit", BUDGET_LIMIT)

    # 调用 @task，用 .result() 获取结果
    needs_approval = decide_approval(budget_total, budget_limit).result()

    return {"needs_approval": needs_approval}


def build_budget_subgraph():
    """返回编译后的 functional 子图（可直接作为节点加入父图）。"""
    return evaluate_budget
