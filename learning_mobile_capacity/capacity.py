"""联合运力决策与服务承诺核算。

资源约定沿用排期规则：
- 每辆教学车每日可用工时固定，路程占用与现场服务一起消耗工时；
- 每次开班需要具备对应科目的随班教师与设备，缺少资格的社区不能排课；
- 单车每日沿一条线路顺序访问，同一社区同一天最多访问一次，访问时可连开多班；
- 已确认的最低服务次数是硬约束，任何预测不利都不能静默取消。

决策输出不制造“带成功标记的空计划”：当最低承诺本身互相冲突，或所有可行
安排都达不到约定可靠度时，状态显式标记为不可承诺，并给出可调整范围与
风险来源（需求本身 / 资料不足 / 路线资源冲突）。
"""
from __future__ import annotations

import json
from pathlib import Path

from .forecast import (breach_envelope, horizon_moments, normal_excess)

# 可靠性目标的默认值：规划周期内无社区短供的概率下界不低于该值
DEFAULT_TARGET_RELIABILITY = 0.85


# ---------------------------------------------------------------------------
# 资源配置
# ---------------------------------------------------------------------------
def load_resources(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"day_minutes", "service_minutes", "session_capacity",
                "teachers", "equipment", "travel_minutes"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"资源配置缺少字段: {', '.join(sorted(missing))}")
    travel = {}
    for pair, minutes in data["travel_minutes"].items():
        endpoints = tuple(sorted(pair.split("/")))
        if len(endpoints) != 2:
            raise ValueError(f"路程键必须形如 '社区甲/社区乙': {pair}")
        travel[endpoints] = int(minutes)
    return {
        "day_minutes": int(data["day_minutes"]),
        "service_minutes": int(data["service_minutes"]),
        "session_capacity": int(data["session_capacity"]),
        "teachers": {k: set(v) for k, v in data["teachers"].items()},
        "equipment": {k: set(v) for k, v in data["equipment"].items()},
        "travel_minutes": travel,
    }


def _travel_time(resources, a, b):
    if a == b:
        return 0
    return resources["travel_minutes"][tuple(sorted((a, b)))]


def _check_qualifications(resources, communities):
    """每次开班都必须有对应科目的教师与设备随班配套（资格校验）。

    单教学车每天沿一条线路顺序访问各点，同一时刻只服务一个社区，
    因此不存在同一教师/设备在两个社区“同时”被占用；资源约束体现在
    随班资格是否齐备，以及工时/路程是否装得进当天可用分钟数。
    """
    for community in communities:
        if not resources["teachers"].get(community):
            return False, f"社区 {community} 缺少可派教师"
        if not resources["equipment"].get(community):
            return False, f"社区 {community} 缺少随班设备"
    return True, ""


def minimum_days(resources, sessions_by_community):
    """计算落实给定班次组合所需的最少工作日。

    每天把若干社区打包进同一辆车的一条线路：顺序访问，访问时长为
    班次×单班服务分钟，社区之间叠加路程。用回溯寻找可行的最少天数；
    装不进当天可用分钟数的社区只能安排到别的工作日。
    """
    stops = sorted((c, s) for c, s in sessions_by_community.items() if s > 0)
    if not stops:
        return 0, []

    qualified, reason = _check_qualifications(resources, [c for c, _ in stops])
    if not qualified:
        return None, reason

    limit = len(stops)

    def day_duration(day):
        total = 0
        ordered = sorted(day)
        for idx, community in enumerate(ordered):
            total += sessions_by_community[community] * resources["service_minutes"]
            if idx:
                total += _travel_time(resources, ordered[idx - 1], community)
        return total

    def can_fit(day, community):
        # 单线路顺序访问，约束只来自当天工时与路程，枚举访问顺序取最短线路
        candidates = day + [community]
        best = min(day_duration(perm)
                   for perm in _permutations(candidates))
        return best <= resources["day_minutes"]

    def backtrack(index, days):
        if index == len(stops):
            return days
        community, _ = stops[index]
        # 先尝试装入已有工作日，再考虑新开一天；DFS 顺序保证首次成功即最少天数
        for day in days:
            if can_fit(day, community):
                day.append(community)
                found = backtrack(index + 1, days)
                if found is not None:
                    return found
                day.remove(community)
        if len(days) < limit:
            found = backtrack(index + 1, days + [[community]])
            if found is not None:
                return found
        return None

    days = backtrack(0, [])
    if days is None:
        return None, "所有社区各占一天仍无法在当天可用工时内完成"
    return len(days), days


def _permutations(items):
    if len(items) <= 1:
        yield tuple(items)
        return
    for idx, item in enumerate(items):
        rest = items[:idx] + items[idx + 1:]
        for tail in _permutations(rest):
            yield (item,) + tail


# ---------------------------------------------------------------------------
# 可行组合枚举
# ---------------------------------------------------------------------------
def _ranges(ranges):
    """每个社区在各自的闭区间内独立取值的笛卡尔积。

    ``ranges[i] = (下限, 上限)``，返回元组第 i 个分量与调用方
    ``groups[i]``（已排序）一一对应。
    """
    if not ranges:
        yield ()
        return
    (lo, hi), *rest = ranges
    for value in range(lo, hi + 1):
        for tail in _ranges(rest):
            yield (value,) + tail


def enumerate_plans(forecast, resources, minimum_sessions, available_days,
                    max_extra=3):
    """枚举满足硬约束的全部班次组合，返回带资源核算的计划清单。"""
    groups = sorted(forecast["communities"])
    for group in groups:
        if group not in minimum_sessions:
            raise KeyError(f"缺少社区 {group} 的最低服务承诺")
        if group not in resources["teachers"] or group not in resources["equipment"]:
            raise KeyError(f"资源配置缺少社区 {group} 的教师或设备条目")

    base = tuple(minimum_sessions[g] for g in groups)
    slot_ranges = [(base[i], base[i] + max_extra) for i in range(len(groups))]
    plans = []
    for composition in _ranges(slot_ranges):
        sessions = dict(zip(groups, composition))
        days_needed, detail = minimum_days(resources, sessions)
        if days_needed is None:
            blockers = [detail]
        elif days_needed > available_days:
            blockers = [f"最少需要 {days_needed} 个工作日，"
                        f"超出可用的 {available_days} 天"]
        else:
            blockers = []
        plan = {
            "sessions": sessions,
            "days_needed": days_needed,
            "within_days": not blockers,
            "blockers": blockers,
            "slack_days": (available_days - days_needed)
                          if days_needed is not None and not blockers else None,
        }
        plans.append(plan)
    return plans


# ---------------------------------------------------------------------------
# 计划评估与决策
# ---------------------------------------------------------------------------
def evaluate_plan(forecast, resources, plan):
    """对单个可行计划计算运力、可靠性、未覆盖需求与风险归因。"""
    horizon = forecast["horizon"]
    capacity_per_session = resources["session_capacity"]
    capacities = {g: plan["sessions"][g] * capacity_per_session
                  for g in forecast["communities"]}
    envelope = breach_envelope(forecast, capacities)

    point_metrics = {}
    total_expected_uncovered = 0.0
    for group, marginal in forecast["communities"].items():
        mu, sd = horizon_moments(marginal, horizon)
        cap = capacities[group]
        uncovered = normal_excess(mu, sd, cap)
        total_expected_uncovered += uncovered
        point_metrics[group] = {
            "sessions": plan["sessions"][group],
            "capacity": cap,
            "demand_mean": mu,
            "expected_uncovered": uncovered,
            "shortfall_probability": envelope["per_community"][group],
            "flags": list(marginal["flags"]),
        }

    reliability_lower = 1.0 - envelope["breach_upper"]
    reliability_upper = 1.0 - envelope["breach_lower"]
    return {
        **plan,
        "capacities": capacities,
        "point_metrics": point_metrics,
        "expected_uncovered": total_expected_uncovered,
        "breach_lower": envelope["breach_lower"],
        "breach_upper": envelope["breach_upper"],
        "reliability_lower": reliability_lower,
        "reliability_upper": reliability_upper,
        "joint_scenarios": [
            {key: s[key] for key in ("key", "title", "mean", "sd", "q90")}
            for s in forecast["scenarios"]
        ],
        "total_capacity": sum(capacities.values()),
    }


def decide(forecast, resources, minimum_sessions, available_days,
           target_reliability=DEFAULT_TARGET_RELIABILITY, max_extra=3):
    """主决策入口：比较可执行组合，给出承诺或显式报告无法承诺。"""
    plans = enumerate_plans(forecast, resources, minimum_sessions,
                            available_days, max_extra=max_extra)
    feasible = [p for p in plans if p["within_days"]]
    blocked = [p for p in plans if not p["within_days"]]
    data_risks = _data_risks(forecast)

    if not feasible:
        base_sessions = dict(zip(sorted(forecast["communities"]),
                                 (minimum_sessions[g]
                                  for g in sorted(forecast["communities"]))))
        days_needed, blockers = minimum_days(resources, base_sessions)
        return {
            "status": "最低承诺不可行",
            "target_reliability": target_reliability,
            "forecast_as_of": forecast["as_of"],
            "plans": [],
            "recommended": None,
            "risks": data_risks + [{
                "source": "路线资源冲突",
                "detail": blockers or f"可用 {available_days} 个工作日无法容纳最低班次",
            }],
            "tradeoffs": [],
            "adjustable": _infeasible_adjustments(resources, available_days,
                                                  blockers),
            "cannot_promise": [f"最低服务承诺 {minimum_sessions} 无法在 "
                               f"{available_days} 个工作日内落实"],
        }

    evaluated = sorted((evaluate_plan(forecast, resources, p) for p in feasible),
                       key=lambda p: (-p["reliability_lower"],
                                      p["expected_uncovered"],
                                      p["days_needed"]))
    committable = [p for p in evaluated
                   if p["reliability_lower"] >= target_reliability]
    tradeoffs = _tradeoffs(evaluated, forecast, resources)
    resource_risks = _resource_risks(evaluated, blocked, available_days)

    if committable:
        return {
            "status": "可承诺",
            "target_reliability": target_reliability,
            "forecast_as_of": forecast["as_of"],
            "plans": evaluated,
            "recommended": committable[0],
            "risks": data_risks + resource_risks,
            "tradeoffs": tradeoffs,
            "adjustable": _commitment_adjustments(evaluated, target_reliability,
                                                   committable[0]),
            "cannot_promise": [],
        }

    best = evaluated[0]
    return {
        "status": "可靠性不足，不可按约定承诺",
        "target_reliability": target_reliability,
        "forecast_as_of": forecast["as_of"],
        "plans": evaluated,
        "recommended": None,
        "risks": data_risks + resource_risks + [{
            "source": "需求本身",
            "detail": (f"最优安排的可靠性下界仅 {best['reliability_lower']:.2%}，"
                       f"低于约定的 {target_reliability:.0%}；"
                       f"期望未覆盖需求 {best['expected_uncovered']:.1f} 人次"),
        }],
        "tradeoffs": tradeoffs,
        "adjustable": _commitment_adjustments(evaluated, target_reliability),
        "cannot_promise": [
            f"无法在 {available_days} 个工作日内同时保证最低班次与 "
            f"{target_reliability:.0%} 的服务可靠度",
        ],
    }


# ---------------------------------------------------------------------------
# 风险归因与权衡
# ---------------------------------------------------------------------------
def _data_risks(forecast):
    risks = []
    weak = [g for g, m in forecast["communities"].items() if m["flags"]]
    if weak:
        risks.append({
            "source": "资料不足",
            "detail": "观察期不足或方差借用：" + "；".join(
                f"{g}（{'；'.join(forecast['communities'][g]['flags'])}）"
                for g in weak),
        })
    unknown = [p for p in forecast["pairs"] if p["unknown"]]
    if unknown:
        risks.append({
            "source": "资料不足",
            "detail": "以下点位对没有足够共同观察，相依性未知，联合风险以 "
                      "Fréchet 界表示：" + "；".join(
                          f"{p['a']}/{p['b']}（重叠 {p['overlap']} 期）"
                          for p in unknown),
        })
    if forecast["shared_variance"] is not None and forecast["shared_variance"] > 0:
        risks.append({
            "source": "需求本身",
            "detail": (f"共同观察期存在同步起伏，公共因子方差 "
                       f"{forecast['shared_variance']:.4f}，"
                       "节假日或同一活动会让多个社区同时高涨"),
        })
    return risks


def _resource_risks(evaluated, blocked, available_days):
    risks = []
    tightest = min(p["slack_days"] for p in evaluated)
    if tightest <= 0:
        risks.append({
            "source": "路线资源冲突",
            "detail": "最稳档位已用满全部工作日，继续加开只能压缩其他点位或"
                      "产生不可行组合",
        })
    if blocked:
        risks.append({
            "source": "路线资源冲突",
            "detail": f"共 {len(blocked)} 个加开组合因当天工时或路程超限"
                      f"无法在 {available_days} 天内执行",
        })
    return risks


def _tradeoffs(evaluated, forecast, resources):
    """量化“给一个社区加开班次会压缩其他点的可兑现能力”。

    只保留帕累托前沿：以总工作日为成本、可靠性下界与期望未覆盖为收益，
    被其他安排全面支配的组合不进入决策视野。
    """
    frontier = []
    for plan in evaluated:
        cost = plan["days_needed"]
        dominated = False
        for other in evaluated:
            if other is plan:
                continue
            if (other["days_needed"] <= cost
                    and other["reliability_lower"] >= plan["reliability_lower"]
                    and other["expected_uncovered"] <= plan["expected_uncovered"]
                    and (other["days_needed"] < cost
                         or other["reliability_lower"] > plan["reliability_lower"]
                         or other["expected_uncovered"] < plan["expected_uncovered"])):
                dominated = True
                break
        if not dominated:
            frontier.append(plan)

    minimum_plan = min(evaluated, key=lambda p: sum(p["sessions"].values()))
    tradeoffs = []
    for plan in sorted(frontier, key=lambda p: (p["days_needed"],
                                                -p["reliability_lower"])):
        if plan is minimum_plan:
            continue
        added = {g: plan["sessions"][g] - minimum_plan["sessions"][g]
                 for g in plan["sessions"]
                 if plan["sessions"][g] > minimum_plan["sessions"][g]}
        if not added:
            continue
        rel_gain = (plan["reliability_lower"]
                    - minimum_plan["reliability_lower"])
        uncov_gain = (minimum_plan["expected_uncovered"]
                      - plan["expected_uncovered"])
        # 既不实质提高可靠性（<0.5 个百分点）也不减少未覆盖（<0.1 人次）
        # 的加开不构成有效权衡，避免浮点噪声进入报告
        if rel_gain < 0.005 and uncov_gain < 0.1:
            continue
        tradeoffs.append({
            "add_sessions": added,
            "extra_days": plan["days_needed"] - minimum_plan["days_needed"],
            "capacity_gain": plan["total_capacity"]
                             - minimum_plan["total_capacity"],
            "expected_uncovered_reduction": uncov_gain,
            "reliability_lower_change": rel_gain,
            "note": ("加开带来的运力增量只有在当天工时与路程未被其他点占用时"
                     "可兑现；工作日被占满后继续加开只会产生不可行组合"),
        })
    return tradeoffs


def _commitment_adjustments(plans, target_reliability, best=None):
    if not plans:
        return []
    best = best or plans[0]
    return [
        f"全部可行组合的可靠性下界范围 "
        f"{min(p['reliability_lower'] for p in plans):.2%}–"
        f"{max(p['reliability_lower'] for p in plans):.2%}，"
        f"其中不低于 {target_reliability:.0%} 的档位可按约定承诺",
        f"最稳安排为 {best['sessions']}，需 {best['days_needed']} 个工作日，"
        f"期望未覆盖 {best['expected_uncovered']:.1f} 人次",
        f"若需高于 {target_reliability:.0%} 的可靠度，需增加可用工作日、"
        "补充教师/设备或下调最低承诺后重新核算",
    ]


def _infeasible_adjustments(resources, available_days, blockers):
    return [
        f"增加可用工作日（当前 {available_days} 天）或缩短单班服务时长"
        f"（当前 {resources['service_minutes']} 分钟）",
        "调配随班教师/设备以支持更多社区的开班资格",
        "重新协商各社区最低服务次数；具体冲突：" + str(blockers),
    ]
