"""端到端演示脚本（异步执行）。

依次演示 LangGraph 运行时特性：
  1) 流式输出（stream_mode="updates"）+ 并行研究（Send Map-Reduce）
  2) 人工审批（interrupt + Command(resume=)）
  3) 反思循环（审批拒绝 → revise → 回到 plan）
  4) 长期记忆（BaseStore 跨 thread 读写偏好）
  5) 时间旅行（aget_state_history 回看历史状态）
  6) 手动改状态（aupdate_state 注入修正后续跑）
  7) 编译期断点（interrupt_before）

运行前请确保已配置 .env 中的 DEEPSEEK_API_KEY。
"""
from __future__ import annotations

import asyncio
import os

import graph as graph_mod
from store import GLOBAL_STORE, load_user_preferences, save_user_preference, search_users

# 演示用独立 sqlite，避免跨次运行累积
DB = "checkpoints_demo.sqlite"
if os.path.exists(DB):
    os.remove(DB)


async def _print_snapshot(label: str, snapshot) -> None:
    print(f"\n--- {label} ---")
    print(f"  next 节点: {snapshot.next}")
    vals = snapshot.values or {}
    print(f"  destinations: {vals.get('destinations')}")
    print(f"  research_results 数量: {len(vals.get('research_results', []))}")
    print(f"  budget_total: {vals.get('budget_total')} / limit {vals.get('budget_limit')}")
    print(f"  needs_approval: {vals.get('needs_approval')}")
    print(f"  approved: {vals.get('approved')}  iterations: {vals.get('iterations')}")


async def run_flow(app, user_input: dict, thread_id: str, decisions=("", "yes")):
    """异步流式运行；遇 interrupt 用 decisions 队列依次续跑，直到完成。"""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    decision_iter = iter(decisions)
    cur = user_input
    segment = 0

    while True:
        segment += 1
        print(f"\n========== 流式第 {segment} 段 (thread={thread_id}) ==========")
        async for chunk in app.astream(cur, config=config, stream_mode="updates"):
            for node, upd in chunk.items():
                if isinstance(upd, dict):
                    print(f"  • 节点 [{node}] 更新键: {list(upd.keys())}")
                else:
                    print(f"  • 节点 [{node}] -> {upd}")

        snapshot = await app.aget_state(config)

        # 检测 in-node interrupt
        interrupts = []
        for t in snapshot.tasks:
            interrupts.extend(getattr(t, "interrupts", []) or [])

        if interrupts:
            try:
                dec = next(decision_iter)
            except StopIteration:
                print("  ⏸ 预设决策已用尽，停在 interrupt。")
                return snapshot
            print(f"  ⏸ 检测到 interrupt，注入人工决策 resume={dec!r}")
            cur = Command(resume=dec)
            continue

        if snapshot.next:
            cur = None
            continue

        break

    return await app.aget_state(config)


# ---------------------------------------------------------------------------
# 1) 主流程：并行研究 + 审批 + 反思循环
# ---------------------------------------------------------------------------
async def demo_main_flow(app):
    print("\n" + "=" * 70)
    print("演示 1：主流程（并行研究 / 流式 / 审批 / 反思循环）")
    print("=" * 70)

    user_input = {"query": "我想去北京和上海旅游，各玩两天", "budget_limit": 1000.0}
    snapshot = await run_flow(
        app,
        user_input,
        thread_id="trip-1",
        decisions=("no", "yes"),
    )
    await _print_snapshot("主流程最终状态", snapshot)
    print("\n最终行程预览：")
    print((snapshot.values or {}).get("final_itinerary", "（无）"))


# ---------------------------------------------------------------------------
# 2) 长期记忆：跨 thread 写入并验证读取
# ---------------------------------------------------------------------------
async def demo_memory(app):
    print("\n" + "=" * 70)
    print("演示 2：长期记忆（BaseStore，跨 thread）")
    print("=" * 70)

    save_user_preference(GLOBAL_STORE, "demo_user", "style", "省钱自由行")
    save_user_preference(GLOBAL_STORE, "demo_user", "hotel", "民宿")
    prefs = load_user_preferences(GLOBAL_STORE, "demo_user")
    print(f"写入并读回偏好: {prefs}")

    # 演示 BaseStore 搜索
    hits = search_users(GLOBAL_STORE, "style")
    print(f"BaseStore 按关键词搜索命中: {hits}")

    config = {"configurable": {"thread_id": "trip-mem"}}
    async for chunk in app.astream(
        {"query": "去成都玩", "budget_limit": 1000.0},
        config=config,
        stream_mode="updates",
    ):
        for node, upd in chunk.items():
            if node == "understand" and isinstance(upd, dict):
                print(f"understand 读到的偏好: {upd.get('user_preferences')}")
        # 仅观察 understand 的注入即可，不等流程跑完
        break

    print("✓ 跨 thread 记忆已被 understand 节点读取注入。")


# ---------------------------------------------------------------------------
# 3) 时间旅行：aget_state_history 回看历史
# ---------------------------------------------------------------------------
async def demo_time_travel(app):
    print("\n" + "=" * 70)
    print("演示 3：时间旅行（aget_state_history）")
    print("=" * 70)

    config = {"configurable": {"thread_id": "trip-1"}}
    history = [s async for s in app.aget_state_history(config)]
    print(f"历史状态快照数: {len(history)}")
    for i, s in enumerate(history[:8]):
        cid = s.config["configurable"].get("checkpoint_id")
        vals = s.values or {}
        print(
            f"  [{i}] ckpt={cid[:8]} next={s.next} "
            f"approved={vals.get('approved')} budget={vals.get('budget_total')}"
        )

    if len(history) >= 6:
        past = history[5]
        print(f"\n回看到历史检查点 {past.config['configurable']['checkpoint_id'][:8]}：")
        print(f"  当时 next={past.next}, values键={list((past.values or {}).keys())}")


# ---------------------------------------------------------------------------
# 4) 手动改状态：aupdate_state 注入修正后续跑
# ---------------------------------------------------------------------------
async def demo_update_state(app):
    print("\n" + "=" * 70)
    print("演示 4：aupdate_state 手动改状态")
    print("=" * 70)

    config = {"configurable": {"thread_id": "trip-update"}}
    async for _ in app.astream(
        {"query": "去东京旅游", "budget_limit": 1000.0},
        config=config,
        stream_mode="updates",
    ):
        pass
    snap = await app.aget_state(config)
    print(f"当前停在 next={snap.next}；budget_total={snap.values.get('budget_total')}")

    await app.aupdate_state(
        config,
        values={"budget_total": 500.0, "approved": True},
    )
    snap2 = await app.aget_state(config)
    print(f"aupdate_state 后：budget_total={snap2.values.get('budget_total')}, approved={snap2.values.get('approved')}")

    from langgraph.types import Command

    final = await app.ainvoke(Command(resume="yes"), config=config)
    print("✓ 注入状态后流程完成，final_itinerary 长度:",
          len((final or {}).get("final_itinerary", "")))


# ---------------------------------------------------------------------------
# 5) 编译期断点：interrupt_before
# ---------------------------------------------------------------------------
async def demo_interrupt_before(app):
    print("\n" + "=" * 70)
    print("演示 5：编译期断点 interrupt_before=['plan']")
    print("=" * 70)

    config = {"configurable": {"thread_id": "trip-break"}}
    async for chunk in app.astream({"query": "去巴黎", "budget_limit": 1000.0}, config=config, stream_mode="updates"):
        for node, _ in chunk.items():
            print(f"  • 执行 [{node}]")
    snap = await app.aget_state(config)
    print(f"✓ 已停在 next={snap.next}（在 plan 之前）—— interrupt_before 生效")


async def main():
    # 异步检查点（timeout 等特性要求异步执行）
    checkpointer = await graph_mod.make_async_sqlite_checkpointer(DB)
    app = graph_mod.build_graph(
        checkpointer=checkpointer,
        store=GLOBAL_STORE,
        interrupt_before=None,
    )
    # 用于演示 5 的带断点图（共享同一 checkpointer）
    app_break = graph_mod.build_graph(
        checkpointer=checkpointer,
        store=GLOBAL_STORE,
        interrupt_before=["plan"],
    )

    await demo_main_flow(app)
    await demo_memory(app)
    await demo_time_travel(app)
    await demo_update_state(app)
    await demo_interrupt_before(app_break)
    print("\n全部演示完成。")


if __name__ == "__main__":
    asyncio.run(main())
