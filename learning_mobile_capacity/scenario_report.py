"""联合需求情景与运力决策报告（中文输出）。

运行：``python3 -m learning_mobile_capacity.scenario_report``
报告把预测依据、未知依赖、可执行组合、风险来源与可调整范围分开陈述，
当无法按约定承诺时显式说明，不输出带成功标记的空计划。
"""
from __future__ import annotations

import json
from pathlib import Path

from .capacity import decide, load_resources
from .forecast import forecast_demand
from .records import read_records

DATA_DIR = Path(__file__).parent.parent / "data"


def build_report(rows, resources, commitments):
    forecast = forecast_demand(
        rows,
        as_of=commitments["as_of"],
        horizon=commitments["horizon"],
    )
    decision = decide(
        forecast,
        resources,
        minimum_sessions=commitments["minimum_sessions"],
        available_days=commitments["available_days"],
        target_reliability=commitments["target_reliability"],
    )
    return forecast, decision


def render(forecast, decision):
    lines = []
    lines.append("=" * 72)
    lines.append("流动教学车联合需求与运力决策报告")
    lines.append("=" * 72)
    lines.append(f"信息截点：{forecast['as_of']}（仅使用该日期前已收到的记录）；"
                 f"预测周期数：{forecast['horizon']}")

    lines.append("")
    lines.append("一、各社区单点需求预测（含不确定性与资料标记）")
    lines.append("-" * 72)
    header = f"{'社区':<8}{'样本期':>5}{'均值':>8}{'标准差':>8}" \
             f"{'P10':>8}{'P50':>8}{'P90':>8}  资料标记"
    lines.append(header)
    for group, marginal in forecast["communities"].items():
        flags = "；".join(marginal["flags"]) or "充分"
        lines.append(
            f"{group:<8}{marginal['observations']:>5}"
            f"{marginal['mean']:>8.1f}{marginal['sd']:>8.2f}"
            f"{marginal['q10']:>8.1f}{marginal['q50']:>8.1f}"
            f"{marginal['q90']:>8.1f}  {flags}（方差来源："
            f"{marginal['variance_source']}）")

    lines.append("")
    lines.append("二、联合情景的证据基础")
    lines.append("-" * 72)
    if forecast["shared_periods"]:
        lines.append("共同观察期（保留节假日/同一活动导致的同步起伏）：")
        for item in forecast["shared_periods"]:
            lines.append(f"  {item['period']} 公共因子={item['factor']:+.2f}，"
                         f"覆盖：{'、'.join(item['communities'])}")
    else:
        lines.append("不存在两个及以上社区共同观察的周期，无法估计共同起伏。")
    shared_var = forecast["shared_variance"]
    lines.append(f"公共因子方差：{shared_var:.4f}" if shared_var is not None
                 else "公共因子方差：样本不足，无法估计")
    lines.append("点位间相依性：")
    for pair in forecast["pairs"]:
        if pair["unknown"]:
            lines.append(f"  {pair['a']} / {pair['b']}：{pair['evidence']}")
        else:
            lines.append(f"  {pair['a']} / {pair['b']}：相关系数 "
                         f"{pair['correlation']:+.3f}（{pair['evidence']}）")

    lines.append("")
    lines.append("三、联合总需求情景（明确依据与未澄清依赖）")
    lines.append("-" * 72)
    for scenario in forecast["scenarios"]:
        lines.append(f"[{scenario['key']}] {scenario['title']}")
        lines.append(f"  依据：{scenario['basis']}")
        lines.append(f"  均值={scenario['mean']:.1f}，标准差={scenario['sd']:.2f}，"
                     f"P90={scenario['q90']:.1f}")
        for assumption in scenario["assumptions"]:
            lines.append(f"  依赖假设：{assumption}")

    lines.append("")
    lines.append("四、运力决策")
    lines.append("-" * 72)
    lines.append(f"决策状态：{decision['status']}")
    lines.append(f"约定可靠性目标：{decision['target_reliability']:.0%}"
                 "（按未知相依结构下的可靠性下界判定）")
    lines.append(f"可比的可执行组合数：{len(decision['plans'])}")
    if decision["recommended"]:
        plan = decision["recommended"]
        lines.append(f"推荐班次组合：{plan['sessions']}")
        lines.append(f"  所需工作日：{plan['days_needed']}，"
                     f"剩余机动日：{plan['slack_days']}")
        lines.append(f"  各班次运力：{plan['capacities']}")
        lines.append(f"  期望未覆盖需求：{plan['expected_uncovered']:.2f} 人次")
        lines.append(f"  可靠性区间：{plan['reliability_lower']:.2%} – "
                     f"{plan['reliability_upper']:.2%}")
        lines.append("  各点短供概率：" + "，".join(
            f"{g} {m['shortfall_probability']:.1%}"
            for g, m in plan["point_metrics"].items()))
    else:
        lines.append("不存在满足全部约定的推荐组合，以下为最接近的可行安排：")
        if decision["plans"]:
            plan = decision["plans"][0]
            lines.append(f"  班次：{plan['sessions']}，工作日 "
                         f"{plan['days_needed']}，可靠性下界 "
                         f"{plan['reliability_lower']:.2%}，"
                         f"期望未覆盖 {plan['expected_uncovered']:.1f} 人次")

    lines.append("")
    lines.append("五、风险登记（可追溯到需求本身 / 资料不足 / 路线资源冲突）")
    lines.append("-" * 72)
    if decision["risks"]:
        for risk in decision["risks"]:
            lines.append(f"[{risk['source']}] {risk['detail']}")
    else:
        lines.append("无登记风险。")

    lines.append("")
    lines.append("六、组合间权衡（一个社区加开对其他点可兑现能力的影响）")
    lines.append("-" * 72)
    if decision["tradeoffs"]:
        for trade in decision["tradeoffs"]:
            lines.append(
                f"加开 {trade['add_sessions']}：多占用 {trade['extra_days']} 个工作日，"
                f"运力增量 {trade['capacity_gain']} 人次，"
                f"期望未覆盖减少 {trade['expected_uncovered_reduction']:.2f}，"
                f"可靠性下界变化 {trade['reliability_lower_change']:+.2%}")
            lines.append(f"  说明：{trade['note']}")
    else:
        lines.append("没有可比较的加开空间。")

    lines.append("")
    lines.append("七、无法作出的承诺与可调整范围")
    lines.append("-" * 72)
    if decision["cannot_promise"]:
        for item in decision["cannot_promise"]:
            lines.append(f"无法承诺：{item}")
    else:
        lines.append("当前推荐组合可在约定可靠度内承诺最低服务次数。")
    for item in decision["adjustable"]:
        lines.append(f"可调整：{item}")

    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    rows = read_records(DATA_DIR / "demand_history.csv")
    resources = load_resources(DATA_DIR / "resources.json")
    commitments = json.loads(
        (DATA_DIR / "commitments.json").read_text(encoding="utf-8"))
    forecast, decision = build_report(rows, resources, commitments)
    print(render(forecast, decision))


if __name__ == "__main__":
    main()
