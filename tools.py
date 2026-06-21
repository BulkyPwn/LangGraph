"""自定义工具：天气查询 / 旅游搜索 / 预算计算。

演示特性：
- @tool 装饰器自定义工具（LangChain 工具协议）
- 工具会被 ToolNode 执行、并由 tools_condition 路由
- 其中 budget_calculator 故意做成“偶发失败”，便于演示 RetryPolicy
"""
from __future__ import annotations

import random

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询某城市当前天气（演示用，返回模拟数据）。"""
    # 真实场景可接入天气 API，此处用确定性模拟以便离线演示
    fake = {
        "北京": "晴 26°C",
        "上海": "多云 28°C",
        "成都": "小雨 22°C",
        "东京": "晴 24°C",
        "巴黎": "阴 18°C",
    }
    return f"{city} 天气：{fake.get(city, '晴 25°C')}"


@tool
def search_attractions(city: str) -> str:
    """搜索某城市热门景点（演示用，返回模拟数据）。"""
    data = {
        "北京": "故宫、长城、颐和园",
        "上海": "外滩、迪士尼、豫园",
        "成都": "大熊猫基地、宽窄巷子、都江堰",
        "东京": "浅草寺、涩谷、东京塔",
        "巴黎": "埃菲尔铁塔、卢浮宫、凯旋门",
    }
    return f"{city} 推荐景点：{data.get(city, '市中心观光区')}"


@tool
def budget_calculator(items: list[float]) -> str:
    """对一组花费求和。

    演示特性：以 30% 概率抛错，用于在 graph.py 中给该工具所在节点
    配置 RetryPolicy，体现“失败自动重试”。
    """
    if random.random() < 0.3:
        raise RuntimeError("预算服务暂时不可用（模拟失败，触发 RetryPolicy）")
    total = sum(items)
    return f"各项花费 {items} 合计 = {total} 元"


# 供 ToolNode 使用的工具集合
ALL_TOOLS = [get_weather, search_attractions, budget_calculator]
