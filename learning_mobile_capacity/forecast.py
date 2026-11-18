"""联合需求预测。

设计原则（对应运营约束）：

1. 严格信息截点：只使用 ``as_of`` 之前已经收到的记录；可选的 ``received_at``
   字段表示记录实际到达时间，缺省时视为该周期结束时才可用。任何排期日之后
   才出现的信息都不会进入预测，杜绝把开班后的实际到场回填进当时预测。
2. 保留共同起伏：在多个社区共同观察到的周期上估计公共因子（节假日、同一活动
   造成的同步涨跌），并在有足够重叠期时估计点位间协方差。
3. 不编造相关性：没有共同观察资料的点位对，相关性标记为“未知”，既不假设独立
   也不伪造相关系数，而是用 Fréchet–Hoeffding 界给出联合结果的稳健包络。
4. 所有输出同时携带不确定性与证据说明，情景必须写清依据与尚未澄清的依赖。

仅使用 Python 标准库。
"""
from __future__ import annotations

import math
from collections import defaultdict

# 单个社区至少需要多少期观察才视为资料充分
MIN_OBS = 3
# 点位对至少需要多少期共同观察才允许给出经验相关性
MIN_OVERLAP = 2
# 小样本协方差向 0 收缩的强度（lambda = n / (n + K)）
SHRINKAGE_K = 3.0


# ---------------------------------------------------------------------------
# 标准正态分布工具（无第三方依赖）
# ---------------------------------------------------------------------------
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def norm_ppf(p: float) -> float:
    """标准正态分位数，Acklam 算法 + 一次 Halley 校正。"""
    if not 0.0 < p < 1.0:
        raise ValueError("分位概率必须位于 (0, 1) 区间")
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    # Halley 校正，把尾部误差压到机器精度量级
    error = norm_cdf(x) - p
    u = error * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


def normal_excess(mu: float, sd: float, cap: float) -> float:
    """E[(X - cap)^+]，X 服从 N(mu, sd^2) 的正态缺货期望。"""
    if sd <= 0.0:
        return max(0.0, mu - cap)
    z = (cap - mu) / sd
    return sd * (norm_pdf(z) - z * (1.0 - norm_cdf(z)))


# ---------------------------------------------------------------------------
# 信息截点与历史序列
# ---------------------------------------------------------------------------
def _cutoff_key(stamp: str):
    """把日期戳规范成可比较的 (年, 月, 日) 整数元组，缺省日按 1 补齐。

    截点比较只针对记录到达时间（received_at），周期标签（period）只用于
    标识观察属于哪一期，不参与截点判断，避免 'YYYY-MM' 与 'YYYY-MM-DD'
    两种粒度直接做字符串比较造成误判。
    """
    parts = stamp.split("-")
    if not 1 <= len(parts) <= 3 or not all(parts):
        raise ValueError(f"无法解析的日期戳: {stamp!r}")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        raise ValueError(f"无法解析的日期戳: {stamp!r}") from None
    while len(numbers) < 3:
        numbers.append(1)
    return tuple(numbers)


def build_history(rows, as_of: str):
    """按信息截点过滤记录，整理成 ``{社区: [(周期, 需求), ...]}``。

    ``rows`` 至少含 period/group/value 字段；可选 ``received_at`` 表示记录
    到达时间，缺省时视为该周期首日可用。截点规则是严格早于：到达时间
    不早于 ``as_of`` 的记录一律排除。同一社区同一周期若有多条记录，
    取算术均值（与既有汇总口径一致）。
    """
    if not isinstance(as_of, str) or not as_of:
        raise ValueError("as_of 必须是非空日期字符串")
    cutoff = _cutoff_key(as_of)
    raw = defaultdict(lambda: defaultdict(list))
    for row in rows:
        period = str(row["period"])
        group = str(row["group"])
        received = str(row.get("received_at") or period)
        if _cutoff_key(received) >= cutoff:
            # 排期日当天或之后才收到的信息（例如开班后的实际到场）不得进入预测
            continue
        raw[group][period].append(float(row["value"]))
    history = {}
    for group, buckets in raw.items():
        history[group] = sorted((period, sum(values) / len(values))
                                for period, values in buckets.items())
    if not history:
        raise ValueError("信息截点之前没有任何观察记录，无法预测")
    return history


def _sample_mean_var(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, None
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, variance


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def forecast_demand(rows, as_of: str, horizon: int = 1,
                    min_obs: int = MIN_OBS, min_overlap: int = MIN_OVERLAP):
    """生成带证据说明的单点预测与联合情景。

    返回字典包含：
    - ``as_of`` / ``horizon``：信息截点与预测覆盖的周期数；
    - ``communities``：各点均值、方差来源、分位数与资料充分性标记；
    - ``shared_periods``：共同观察期上的公共因子（同步涨跌）；
    - ``pairs``：点位对的重叠期、经验相关性或“未知”标记；
    - ``scenarios``：联合总需求情景，每个情景写明依据与未澄清依赖。
    """
    if horizon < 1:
        raise ValueError("horizon 必须 >= 1")
    history = build_history(rows, as_of)

    # 合并样本方差，供观察期不足的社区借用（借用本身计入资料风险）
    pooled_values = [v for series in history.values() for _, v in series]
    _, pooled_var = _sample_mean_var(pooled_values)
    if pooled_var is None:
        pooled_var = 0.0

    communities = {}
    for group, series in sorted(history.items()):
        values = [v for _, v in series]
        mean, variance = _sample_mean_var(values)
        flags = []
        if variance is None:
            variance = pooled_var
            source = "合并借用（本点不足两期）"
            flags.append("方差借自合并样本，本点离散度无法直接验证")
        else:
            source = "本点样本"
        if len(values) < min_obs:
            flags.append(f"观察期不足（{len(values)} < {min_obs}）")
        sd = math.sqrt(max(0.0, variance))
        communities[group] = {
            "community": group,
            "observations": len(values),
            "periods": [t for t, _ in series],
            "values": values,
            "mean": mean,
            "variance": variance,
            "sd": sd,
            "variance_source": source,
            "q10": max(0.0, mean + sd * norm_ppf(0.10)),
            "q50": max(0.0, mean + sd * norm_ppf(0.50)),
            "q90": max(0.0, mean + sd * norm_ppf(0.90)),
            "flags": flags,
        }

    shared = _shared_factor(history, communities)
    pairs = _pair_evidence(history, communities, min_overlap)
    scenarios = _joint_scenarios(communities, pairs, horizon)
    return {
        "as_of": as_of,
        "horizon": horizon,
        "communities": communities,
        "shared_periods": shared["periods"],
        "shared_variance": shared["variance"],
        "pairs": pairs,
        "scenarios": scenarios,
    }


def _shared_factor(history, communities):
    """共同观察期上的公共因子：当期各社区残差的横截面均值。"""
    periods = defaultdict(list)
    for group, series in history.items():
        mean = communities[group]["mean"]
        for period, value in series:
            periods[period].append((group, value - mean))
    shared_periods = []
    factors = []
    for period in sorted(periods):
        members = periods[period]
        if len(members) < 2:
            continue
        factor = sum(residual for _, residual in members) / len(members)
        shared_periods.append({
            "period": period,
            "factor": factor,
            "communities": [group for group, _ in members],
        })
        factors.append(factor)
    variance = None
    if len(factors) >= 2:
        mean_factor = sum(factors) / len(factors)
        variance = sum((f - mean_factor) ** 2 for f in factors) / (len(factors) - 1)
    return {"periods": shared_periods, "variance": variance}


def _pair_evidence(history, communities, min_overlap):
    groups = sorted(history)
    result = []
    for idx, a in enumerate(groups):
        map_a = dict(history[a])
        for b in groups[idx + 1:]:
            common = sorted(set(map_a) & set(dict(history[b])))
            entry = {
                "a": a, "b": b,
                "overlap": len(common),
                "periods": common,
                "correlation": None,
                "covariance": None,
                "evidence": "",
                "unknown": True,
            }
            if len(common) >= min_overlap:
                residual_a = [map_a[t] - communities[a]["mean"] for t in common]
                residual_b = [dict(history[b])[t] - communities[b]["mean"] for t in common]
                mean_a = sum(residual_a) / len(common)
                mean_b = sum(residual_b) / len(common)
                cov_emp = sum((x - mean_a) * (y - mean_b)
                              for x, y in zip(residual_a, residual_b)) / (len(common) - 1)
                # 小样本收缩，避免两三期重叠就给出接近 ±1 的确定性相关
                lam = len(common) / (len(common) + SHRINKAGE_K)
                cov = lam * cov_emp
                sd_a, sd_b = communities[a]["sd"], communities[b]["sd"]
                corr = cov / (sd_a * sd_b) if sd_a > 0 and sd_b > 0 else 0.0
                entry.update({
                    "covariance": max(-sd_a * sd_b, min(sd_a * sd_b, cov)),
                    "correlation": max(-1.0, min(1.0, corr)),
                    "evidence": f"共同观察 {len(common)} 期，经验协方差经 {lam:.2f} 收缩",
                    "unknown": False,
                })
            else:
                entry["evidence"] = (
                    f"共同观察仅 {len(common)} 期，相关性未知；"
                    "不假设独立，联合结果改用 Fréchet 界包络")
            result.append(entry)
    return result


def _joint_scenarios(communities, pairs, horizon):
    groups = sorted(communities)
    mean_sum = sum(communities[g]["mean"] for g in groups)
    base_var = sum(communities[g]["variance"] for g in groups)

    known_cov = sum(p["covariance"] or 0.0 for p in pairs if not p["unknown"])
    unknown_sigma = sum(communities[p["a"]]["sd"] * communities[p["b"]]["sd"]
                        for p in pairs if p["unknown"])
    unknown_pairs = [(p["a"], p["b"]) for p in pairs if p["unknown"]]

    def pack(key, title, basis, variance, assumptions, comonotone=False):
        variance = max(0.0, variance)
        mu = horizon * mean_sum
        sd = math.sqrt(horizon * variance)
        scenario = {
            "key": key,
            "title": title,
            "basis": basis,
            "mean": mu,
            "sd": sd,
            "q90": mu + sd * norm_ppf(0.90),
            "assumptions": list(assumptions),
            "unknown_pairs": unknown_pairs,
        }
        if comonotone:
            # 同单调和的分位数等于边际分位数之和，这是任意相依结构下的严格上界
            scenario["comonotone_q90"] = horizon * sum(
                communities[g]["q90"] for g in groups)
        return scenario

    scenarios = []
    if not unknown_pairs:
        evidence_basis = "所有点位对均有共同观察，联合方差完全来自经验协方差"
    else:
        evidence_basis = (
            "已知点位对采用收缩后的经验协方差；未知点位对暂置零，"
            "结论仅为部分证据下的参考，不是完整联合分布")
    scenarios.append(pack(
        "evidence", "证据情景", evidence_basis,
        base_var + 2.0 * known_cov,
        [] if not unknown_pairs else [f"{a}/{b} 相依性未知，按零协方差占位"
                                      for a, b in unknown_pairs]))

    scenarios.append(pack(
        "independent", "独立假设情景",
        "假设各社区需求相互独立；独立是建模假设而非数据证据，"
        "会低估同时高涨的联合风险",
        base_var + 2.0 * known_cov,
        [f"假设 {a} 与 {b} 独立" for a, b in unknown_pairs]))

    scenarios.append(pack(
        "joint_high", "共同高涨上界（同单调）",
        "未知点位对取完全正相依，是 Fréchet–Hoeffding 上界；"
        "对应节假日/同一活动让所有社区同时迎来高峰",
        base_var + 2.0 * known_cov + 2.0 * unknown_sigma,
        [f"未知对 {a}/{b} 取相关系数 +1 的保守上界" for a, b in unknown_pairs],
        comonotone=True))

    scenarios.append(pack(
        "joint_low", "共同低潮下界（反单调）",
        "未知点位对取完全负相依，是方差层面的稳健下界；"
        "用于说明最好情况下联合需求能低到什么程度",
        base_var + 2.0 * known_cov - 2.0 * unknown_sigma,
        [f"未知对 {a}/{b} 取相关系数 -1 的乐观下界" for a, b in unknown_pairs]))
    return scenarios


# ---------------------------------------------------------------------------
# 计划评估时复用的边际/联合概率工具
# ---------------------------------------------------------------------------
def horizon_moments(marginal, horizon):
    mu = horizon * marginal["mean"]
    sd = math.sqrt(horizon) * marginal["sd"]
    return mu, sd


def shortfall_probability(marginal, cap, horizon):
    """单点 P(需求 > 运力)。只依赖边际分布，与点位间相关性无关。"""
    mu, sd = horizon_moments(marginal, horizon)
    if sd <= 0.0:
        return 1.0 if mu > cap else 0.0
    return 1.0 - norm_cdf((cap - mu) / sd)


def breach_envelope(forecast, capacities):
    """至少一个社区短供的概率包络，任意相依结构下严格成立。

    对事件 E_i = {X_i > c_i}，由并集界：
        max_i P(E_i) <= P(并集 E_i) <= min(1, sum_i P(E_i))
    相关性未知时这是可兑现的最严谨表述，不需要伪造联合分布。
    """
    horizon = forecast["horizon"]
    per_point = {}
    for group, marginal in forecast["communities"].items():
        if group not in capacities:
            raise KeyError(f"缺少社区 {group} 的运力安排")
        per_point[group] = shortfall_probability(marginal, capacities[group], horizon)
    return {
        "per_community": per_point,
        "breach_lower": max(per_point.values()),
        "breach_upper": min(1.0, sum(per_point.values())),
    }
