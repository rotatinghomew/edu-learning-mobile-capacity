import unittest

from learning_mobile_capacity.forecast import (breach_envelope,
                                               build_history,
                                               forecast_demand, norm_cdf,
                                               norm_ppf, shortfall_probability)


def rows(*tuples):
    return [{"period": p, "group": g, "value": v,
             "received_at": r or p} for p, g, v, r in tuples]


COMMON_HISTORY = rows(
    # 甲、乙四期完全同步，共同观察证据充分
    ("2026-01", "甲", 10.0, None), ("2026-02", "甲", 12.0, None),
    ("2026-03", "甲", 11.0, None), ("2026-04", "甲", 13.0, None),
    ("2026-01", "乙", 20.0, None), ("2026-02", "乙", 22.0, None),
    ("2026-03", "乙", 21.0, None), ("2026-04", "乙", 23.0, None),
    # 丙只有一期：方差必须借用，且与任何点位的重叠都不足以估计相关性
    ("2026-04", "丙", 6.0, None),
)


class ForecastCutoffTests(unittest.TestCase):
    def test_rows_received_on_or_after_as_of_are_excluded(self):
        data = rows(
            ("2026-05", "甲", 10.0, "2026-05-10"),
            ("2026-06", "甲", 99.0, "2026-07-01"),   # 截点当天到达，排除
            ("2026-07", "甲", 88.0, "2026-08-01"),   # 开班后数据，排除
        )
        history = build_history(data, "2026-07-01")
        self.assertEqual(history, {"甲": [("2026-05", 10.0)]})

    def test_month_period_without_received_at_uses_month_granularity(self):
        # 周期 2026-07 缺省视为 7 月 1 日到达，不得因字符串比较误入截点之前
        data = [{"period": "2026-06", "group": "甲", "value": 10.0},
                {"period": "2026-07", "group": "甲", "value": 99.0}]
        history = build_history(data, "2026-07-01")
        self.assertEqual(history, {"甲": [("2026-06", 10.0)]})

    def test_malformed_stamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "日期戳"):
            build_history(rows(("2026/05", "甲", 1.0, "2026/05/01")),
                          "2026-07-01")

    def test_forecast_never_uses_future_observations(self):
        data = COMMON_HISTORY + rows(
            ("2026-07", "甲", 500.0, "2026-07-20"),
            ("2026-07", "乙", 500.0, "2026-07-20"),
        )
        before = forecast_demand(COMMON_HISTORY, "2026-05-01")
        after = forecast_demand(data, "2026-05-01")
        self.assertEqual(before["communities"]["甲"]["mean"],
                         after["communities"]["甲"]["mean"])
        self.assertEqual(after["communities"]["甲"]["observations"], 4)

    def test_empty_history_after_cutoff_raises(self):
        with self.assertRaisesRegex(ValueError, "没有任何观察"):
            forecast_demand(rows(("2026-08", "甲", 1.0, "2026-08-01")),
                            "2026-07-01")


class JointEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.forecast = forecast_demand(COMMON_HISTORY, "2026-05-01")

    def test_shared_periods_keep_common_swings(self):
        periods = {item["period"] for item in self.forecast["shared_periods"]}
        # 只有甲、乙共同出现的周期才进入公共因子（丙的单期无法估计因子方差）
        self.assertEqual(periods, {"2026-01", "2026-02", "2026-03", "2026-04"})
        self.assertIsNotNone(self.forecast["shared_variance"])

    def test_known_pair_gets_shrunk_correlation(self):
        pair = next(p for p in self.forecast["pairs"]
                    if {p["a"], p["b"]} == {"甲", "乙"})
        self.assertFalse(pair["unknown"])
        self.assertIsNotNone(pair["correlation"])
        # 完全同步的经验相关为 1，经小样本收缩后严格小于 1
        self.assertLess(pair["correlation"], 1.0)
        self.assertGreater(pair["correlation"], 0.5)

    def test_pair_without_overlap_is_unknown_not_independent(self):
        pair = next(p for p in self.forecast["pairs"] if p["a"] == "丙"
                    and p["b"] == "乙")
        self.assertTrue(pair["unknown"])
        self.assertIsNone(pair["correlation"])
        self.assertIn("Fréchet", pair["evidence"])

    def test_frechet_brackets_order(self):
        scenarios = {s["key"]: s for s in self.forecast["scenarios"]}
        self.assertLessEqual(scenarios["joint_low"]["sd"],
                             scenarios["evidence"]["sd"])
        self.assertGreater(scenarios["joint_high"]["sd"],
                           scenarios["evidence"]["sd"])
        self.assertGreater(scenarios["joint_high"]["q90"],
                           scenarios["independent"]["q90"])
        # 同单调 P90 必须等于各边际 P90 之和（严格 Fréchet 上界）
        marginal_sum = sum(m["q90"]
                           for m in self.forecast["communities"].values())
        self.assertAlmostEqual(scenarios["joint_high"]["comonotone_q90"],
                               marginal_sum)

    def test_weak_community_is_flagged_and_uses_pooled_variance(self):
        marginal = self.forecast["communities"]["丙"]
        # 单期观察无法估计本点方差，退回合并样本并显式标记
        self.assertIn("合并借用", marginal["variance_source"])
        self.assertTrue(any("观察期不足" in f for f in marginal["flags"]))


class ProbabilityToolTests(unittest.TestCase):
    def test_norm_ppf_inverts_cdf(self):
        for p in (0.01, 0.1, 0.5, 0.9, 0.99):
            self.assertAlmostEqual(norm_cdf(norm_ppf(p)), p, places=10)

    def test_breach_envelope_is_valid_under_any_dependence(self):
        data = rows(
            ("2026-01", "甲", 10.0, None), ("2026-02", "甲", 12.0, None),
            ("2026-03", "甲", 11.0, None),
            ("2026-01", "乙", 20.0, None), ("2026-02", "乙", 22.0, None),
            ("2026-03", "乙", 21.0, None),
        )
        forecast = forecast_demand(data, "2026-04-01")
        envelope = breach_envelope(forecast, {"甲": 11, "乙": 21})
        per_point = envelope["per_community"].values()
        self.assertGreaterEqual(envelope["breach_lower"], max(per_point))
        self.assertLessEqual(envelope["breach_lower"],
                             envelope["breach_upper"] + 1e-12)
        self.assertLessEqual(envelope["breach_upper"], 1.0 + 1e-12)

    def test_shortfall_probability_monotone_in_capacity(self):
        forecast = forecast_demand(COMMON_HISTORY, "2026-05-01")
        marginal = forecast["communities"]["甲"]
        probs = [shortfall_probability(marginal, cap, 1)
                 for cap in (5, 10, 15, 30)]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_horizon_scales_uncertainty(self):
        from learning_mobile_capacity.forecast import horizon_moments
        forecast = forecast_demand(COMMON_HISTORY, "2026-05-01", horizon=4)
        marginal = forecast["communities"]["甲"]
        mu, sd = horizon_moments(marginal, 4)
        self.assertAlmostEqual(mu, 4 * marginal["mean"])
        self.assertAlmostEqual(sd, 2 * marginal["sd"])


if __name__ == "__main__":
    unittest.main()
