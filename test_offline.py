"""离线冒烟测试：用 Mock LLM 验证图结构/控制流（无需 API Key）。异步执行。"""
import asyncio
import os

from langchain_core.messages import AIMessage

import config
import graph as graph_mod
import nodes
import subgraphs.research as research_mod
from langgraph.types import Command

# 每次运行前清除旧检查点文件，避免持久化累积导致 research_results 数量异常
DB = "checkpoints_smoke.sqlite"
if os.path.exists(DB):
    os.remove(DB)


class FakeLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages, *a, **k):
        return AIMessage(content='["北京","上海"]')

    async def ainvoke(self, messages, *a, **k):
        # plan：不含 tool_calls，从而走 evaluate_budget 分支
        return AIMessage(content="mock 行程计划")


async def main():
    config.make_llm = lambda *a, **k: FakeLLM()
    nodes.make_llm = lambda *a, **k: FakeLLM()
    research_mod.make_llm = lambda *a, **k: FakeLLM()

    checkpointer = await graph_mod.make_async_sqlite_checkpointer(DB)
    app = graph_mod.build_graph(
        checkpointer=checkpointer,
        store=graph_mod.GLOBAL_STORE,
    )
    cfg = {"configurable": {"thread_id": "smoke"}}

    print("== 第一段：跑到 approval 的 interrupt ==")
    async for chunk in app.astream(
        {"query": "去北京和上海", "budget_limit": 1000.0}, config=cfg, stream_mode="updates"
    ):
        for node, upd in chunk.items():
            keys = list(upd.keys()) if isinstance(upd, dict) else upd
            print(f"  [{node}] -> {keys}")

    # AsyncSqliteSaver 须用异步 aget_state
    snap = await app.aget_state(cfg)
    print(f"\n停于 next={snap.next}")
    print(f"research_results 数量={len(snap.values.get('research_results', []))}")
    print(f"budget_total={snap.values.get('budget_total')} needs_approval={snap.values.get('needs_approval')}")
    has_interrupt = any(getattr(t, "interrupts", None) for t in snap.tasks)
    print(f"存在 interrupt={has_interrupt}")

    print("\n== 第二段：resume='no' → revise → 回 plan → 再到 approval ==")
    async for chunk in app.astream(Command(resume="no"), config=cfg, stream_mode="updates"):
        for node, upd in chunk.items():
            keys = list(upd.keys()) if isinstance(upd, dict) else upd
            print(f"  [{node}] -> {keys}")
    snap2 = await app.aget_state(cfg)
    print(f"停于 next={snap2.next} iterations={snap2.values.get('iterations')}")

    print("\n== 第三段：resume='yes' → finalize → END ==")
    final = await app.ainvoke(Command(resume="yes"), config=cfg)
    print(f"final_itinerary 长度={len((final or {}).get('final_itinerary', ''))}")
    print("\n离线冒烟测试通过 ✓")


if __name__ == "__main__":
    asyncio.run(main())
