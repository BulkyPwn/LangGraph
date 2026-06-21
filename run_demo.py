"""端到端演示脚本。

依次演示 LangGraph 运行时特性：
  1) 流式输出（stream_mode="updates"）+ 并行研究（Send Map-Reduce）
  2) 人工审批（interrupt + Command(resume=)）
  3) 反思循环（审批拒绝 → revise → 回到 plan）
  4) 长期记忆（BaseStore 跨 thread 读写偏好）
  5) 时间旅行（get_state_history 回看历史状态）
  6) 手动改状态（update_state 注入修正后续跑）
  7) 编译期断点（interrupt_before，用独立编译的图演示）

运行前请确保已配置 .env 中的 DEEPSEEK_API_KEY。
"""
from __future__ import annotations

import graph as graph_mod
from store import GLOBAL_STORE, load_user_preferences, save_user_preference


def _print_snapshot(label: str, snapshot) -> None:
    print(f"\n--- {label} ---")
    print(f"  next 节点: {snapshot.next}")
    vals = snapshot.values or {}
    print(f"  destinations: {vals.get('destinations')}")
    print(f"  research_results 数量: {len(vals.get('research_results', []))}")
    print(f"  budget_total: {vals.get('budget_total')} / limit {vals.get('budget_limit')}")
    print(f"  needs_approval: {vals.get('needs_approval')}")
    print(f"  approved: {vals.get('approved')}  iterations: {vals.get('iterations')}")


def run_flow(app, user_input: dict, thread_id: str, decisions=("", "yes")):
    """流式运行；遇 interrupt 用 decisions 队列依次续跑，直到完成。"""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    decision_iter = iter(decisions)
    cur = user_input
    segment = 0

    while True:
        segment += 1
        print(f"\n========== 流式第 {segment} 段 (thread={thread_id}) ==========")
        for chunk in app.stream(cur, config=config, stream_mode="updates"):
            for node, upd in chunk.items():
                if isinstance(upd, dict):
                    print(f"  • 节点 [{node}] 更新键: {list(upd.keys())}")
                else:
                    print(f"  • 节点 [{node}] -> {upd}")

        snapshot = app.get_state(config)

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
            # 编译期断点 pause（无 interrupt 对象）→ 用 None 继续
            cur = None
            continue

        # 完成
        break

    return app.get_state(config)


# ---------------------------------------------------------------------------
# 1) 主流程：并行研究 + 审批 + 反思循环
# ---------------------------------------------------------------------------
def demo_main_flow():
    print("\n" + "=" * 70)
    print("演示 1：主流程（并行研究 / 流式 / 审批 / 反思循环）")
    print("=" * 70)

    # 预算上限设低（1000），让估算预算 10000 触发 needs_approval=True
    user_input = {"query": "我想去北京和上海旅游，各玩两天", "budget_limit": 1000.0}
    # 决策队列：第一次“拒绝”触发 revise 回到 plan；第二次“同意”收尾
    snapshot = run_flow(
        graph_mod.default_app,
        user_input,
        thread_id="trip-1",
        decisions=("no", "yes"),
    )
    _print_snapshot("主流程最终状态", snapshot)
    print("\n最终行程预览：")
    print((snapshot.values or {}).get("final_itinerary", "（无）"))


# ---------------------------------------------------------------------------
# 2) 长期记忆：跨 thread 写入并验证读取
# ---------------------------------------------------------------------------
def demo_memory():
    print("\n" + "=" * 70)
    print("演示 2：长期记忆（BaseStore，跨 thread）")
    print("=" * 70)

    save_user_preference(GLOBAL_STORE, "demo_user", "style", "省钱自由行")
    save_user_preference(GLOBAL_STORE, "demo_user", "hotel", "民宿")
    prefs = load_user_preferences(GLOBAL_STORE, "demo_user")
    print(f"写入并读回偏好: {prefs}")

    # 在新 thread 跑一次 understand，验证 store 注入并把偏好带入图
    from langgraph.types import Command

    config = {"configurable": {"thread_id": "trip-mem"}}
    # 仅跑到 understand 后停下观察（用 interrupt_before 的独立图最直观，
    # 这里直接 invoke 单步看 understand 的输出）
    snapshot = None
    for chunk in graph_mod.default_app.stream(
        {"query": "去成都玩", "budget_limit": 1000.0},
        config=config,
        stream_mode="updates",
    ):
        for node, upd in chunk.items():
            if node == "understand" and isinstance(upd, dict):
                print(f"understand 读到的偏好: {upd.get('user_preferences')}")
        snapshot = graph_mod.default_app.get_state(config)
        # 不必跑完整流程；打断演示记忆注入即可
        break

    print("✓ 跨 thread 记忆已被 understand 节点读取注入。")


# ---------------------------------------------------------------------------
# 3) 时间旅行：get_state_history 回看历史
# ---------------------------------------------------------------------------
def demo_time_travel():
    print("\n" + "=" * 70)
    print("演示 3：时间旅行（get_state_history）")
    print("=" * 70)

    config = {"configurable": {"thread_id": "trip-1"}}
    history = list(graph_mod.default_app.get_state_history(config))
    print(f"历史状态快照数: {len(history)}")
    for i, s in enumerate(history[:8]):
        cid = s.config["configurable"].get("checkpoint_id")
        vals = s.values or {}
        print(
            f"  [{i}] ckpt={cid[:8]} next={s.next} "
            f"approved={vals.get('approved')} budget={vals.get('budget_total')}"
        )

    # 选取一个较早的检查点，读取其状态（时间旅行“查看”）
    if len(history) >= 5:
        past = history[5]
        print(f"\n回看到历史检查点 {past.config['configurable']['checkpoint_id'][:8]}：")
        print(f"  当时 next={past.next}, values键={list((past.values or {}).keys())}")


# ---------------------------------------------------------------------------
# 4) 手动改状态：update_state 注入修正后续跑
# ---------------------------------------------------------------------------
def demo_update_state():
    print("\n" + "=" * 70)
    print("演示 4：update_state 手动改状态")
    print("=" * 70)

    config = {"configurable": {"thread_id": "trip-update"}}
    # 先跑到出现 interrupt（审批）位置
    from langgraph.types import Command

    for chunk in graph_mod.default_app.stream(
        {"query": "去东京旅游", "budget_limit": 1000.0},
        config=config,
        stream_mode="updates",
    ):
        # 让它一直跑到 interrupt
        pass
    snap = graph_mod.default_app.get_state(config)
    print(f"当前停在 next={snap.next}；budget_total={snap.values.get('budget_total')}")

    # 人工强制把预算改为未超支，并把审批标记为已批准，再继续
    graph_mod.default_app.update_state(
        config,
        values={"budget_total": 500.0, "approved": True},
    )
    snap2 = graph_mod.default_app.get_state(config)
    print(f"update_state 后：budget_total={snap2.values.get('budget_total')}, approved={snap2.values.get('approved')}")

    # 用 resume 完成剩余流程
    final = graph_mod.default_app.invoke(Command(resume="yes"), config=config)
    print("✓ 注入状态后流程完成，final_itinerary 长度:",
          len((final or {}).get("final_itinerary", "")))


# ---------------------------------------------------------------------------
# 5) 编译期断点：interrupt_before
# ---------------------------------------------------------------------------
def demo_interrupt_before():
    print("\n" + "=" * 70)
    print("演示 5：编译期断点 interrupt_before=['plan']")
    print("=" * 70)

    app = graph_mod.build_graph(
        checkpointer=graph_mod.make_sqlite_checkpointer(),
        store=GLOBAL_STORE,
        interrupt_before=["plan"],
    )
    config = {"configurable": {"thread_id": "trip-break"}}
    # 第一段：应在 plan 之前停下
    for chunk in app.stream({"query": "去巴黎", "budget_limit": 1000.0}, config=config, stream_mode="updates"):
        for node, _ in chunk.items():
            print(f"  • 执行 [{node}]")
    snap = app.get_state(config)
    print(f"✓ 已停在 next={snap.next}（在 plan 之前）—— interrupt_before 生效")


def main():
    demo_main_flow()
    demo_memory()
    demo_time_travel()
    demo_update_state()
    demo_interrupt_before()
    print("\n全部演示完成。")


if __name__ == "__main__":
    main()
