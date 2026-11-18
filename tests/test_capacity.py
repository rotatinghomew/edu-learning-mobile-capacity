import json
import tempfile
import unittest
from pathlib import Path

from learning_mobile_capacity.capacity import (decide, enumerate_plans,
                                               load_resources, minimum_days)
from learning_mobile_capacity.forecast import forecast_demand


def write_resources(tmp, **overrides):
    config = {
        "day_minutes": 480,
        "service_minutes": 90,
        "session_capacity": 10,
        "teachers": {"甲": ["王老师"], "乙": ["李老师"], "丙": ["赵老师"]},
        "equipment": {"甲": ["设备A"], "乙": ["设备B"], "丙": ["设备C"]},
        "travel_minutes": {"甲/乙": 60, "甲/丙": 60, "乙/丙": 60},
    }
    config.update(overrides)
    path = Path(tmp) / "resources.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return load_resources(path)


def steady_rows(group, base, periods=12, swing=1.0):
    # 交替取值：均值 base，样本标准差约为 swing
    values = [base + (swing if i % 2 else -swing) for i in range(periods)]
    return [{"period": f"2025-{i + 1:02d}", "group": group,
             "value": value, "received_at": f"2025-{i + 1:02d}-28"}
            for i, value in enumerate(values)]


def build_forecast():
    rows = steady_rows("甲", 20.0) + steady_rows("乙", 20.0) + [
        # 丙只有一期：方差借用、与其他点位相依性未知
        {"period": "2025-12", "group": "丙", "value": 8.0,
         "received_at": "2025-12-28"},
    ]
    return forecast_demand(rows, "2026-01-01", horizon=1)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.resources = write_resources(tempfile.mkdtemp())

    def test_single_community_duration_includes_service_only(self):
        days, tours = minimum_days(self.resources, {"甲": 2})
        self.assertEqual(days, 1)
        self.assertEqual(tours, [["甲"]])

    def test_two_communities_share_a_day_with_travel_time(self):
        # 各 1 班：90 + 60 + 90 = 240 分钟，一天可完成
        days, tours = minimum_days(self.resources, {"甲": 1, "乙": 1})
        self.assertEqual(days, 1)
        self.assertEqual(len(tours[0]), 2)

    def test_oversized_plan_needs_two_days(self):
        # 各 3 班：270 + 60 + 270 = 600 > 480，必须拆成两天
        days, tours = minimum_days(self.resources, {"甲": 3, "乙": 3})
        self.assertEqual(days, 2)
        self.assertEqual(sorted(len(day) for day in tours), [1, 1])

    def test_missing_qualification_blocks_plan(self):
        resources = write_resources(tempfile.mkdtemp(),
                                    teachers={"甲": ["王老师"],
                                              "乙": ["李老师"], "丙": []})
        days, blockers = minimum_days(resources, {"丙": 1})
        self.assertIsNone(days)
        self.assertIn("教师", blockers)


class EnumerationTests(unittest.TestCase):
    def setUp(self):
        self.forecast = build_forecast()
        self.resources = write_resources(tempfile.mkdtemp())
        self.minimum = {"甲": 1, "乙": 1, "丙": 1}

    def test_minimum_sessions_are_hard_lower_bound(self):
        plans = enumerate_plans(self.forecast, self.resources,
                                self.minimum, available_days=10, max_extra=2)
        for plan in plans:
            for group, floor in self.minimum.items():
                self.assertGreaterEqual(plan["sessions"][group], floor)
        # 区间 [1,3] 三社区笛卡尔积：3^3 = 27 个组合，基础组合必须在其中
        self.assertEqual(len(plans), 27)
        self.assertIn(self.minimum, [p["sessions"] for p in plans])

    def test_blocked_plans_are_reported_not_silently_dropped(self):
        plans = enumerate_plans(self.forecast, self.resources,
                                self.minimum, available_days=1, max_extra=2)
        self.assertTrue(any(p["within_days"] for p in plans))
        blocked = [p for p in plans if not p["within_days"]]
        self.assertTrue(blocked)
        self.assertTrue(all(p["blockers"] for p in blocked))


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.forecast = build_forecast()
        self.resources = write_resources(tempfile.mkdtemp())
        self.minimum = {"甲": 1, "乙": 1, "丙": 1}

    def test_committable_decision_recommends_plan_and_traces_risks(self):
        # 单班运力 10，甲乙均值 20；两班即可覆盖，留出可靠性余量
        decision = decide(self.forecast, self.resources, self.minimum,
                          available_days=5, target_reliability=0.80)
        self.assertEqual(decision["status"], "可承诺")
        self.assertIsNotNone(decision["recommended"])
        self.assertGreaterEqual(
            decision["recommended"]["reliability_lower"], 0.80)
        sources = {risk["source"] for risk in decision["risks"]}
        self.assertIn("资料不足", sources)  # 丙观察期不足 + 相依性未知

    def test_unreachable_target_is_explicitly_refused(self):
        # 只有 1 个工作日：可执行档位无法达到 99% 可靠度
        decision = decide(self.forecast, self.resources, self.minimum,
                          available_days=1, target_reliability=0.99)
        self.assertEqual(decision["status"], "可靠性不足，不可按约定承诺")
        self.assertIsNone(decision["recommended"])
        self.assertTrue(decision["cannot_promise"])
        self.assertTrue(decision["adjustable"])
        self.assertTrue(any(risk["source"] == "需求本身"
                            for risk in decision["risks"]))

    def test_infeasible_minimum_never_returns_empty_success_plan(self):
        resources = write_resources(
            tempfile.mkdtemp(),
            day_minutes=100,  # 单班 90 分钟，任何两社区线路都装不下
            travel_minutes={"甲/乙": 60, "甲/丙": 60, "乙/丙": 60})
        decision = decide(self.forecast, resources, self.minimum,
                          available_days=1, target_reliability=0.5)
        self.assertEqual(decision["status"], "最低承诺不可行")
        self.assertEqual(decision["plans"], [])
        self.assertIsNone(decision["recommended"])
        self.assertTrue(decision["cannot_promise"])
        self.assertTrue(any(risk["source"] == "路线资源冲突"
                            for risk in decision["risks"]))

    def test_tradeoffs_quantify_cross_community_competition(self):
        decision = decide(self.forecast, self.resources, self.minimum,
                          available_days=5, target_reliability=0.0)
        self.assertEqual(decision["status"], "可承诺")
        self.assertTrue(decision["tradeoffs"])
        for trade in decision["tradeoffs"]:
            self.assertGreaterEqual(trade["extra_days"], 0)
            self.assertGreater(trade["capacity_gain"], 0)
            self.assertIn("note", trade)

    def test_more_capacity_never_lowers_reliability(self):
        decision = decide(self.forecast, self.resources, self.minimum,
                          available_days=10, target_reliability=0.0)
        plans = decision["plans"]
        for smaller in plans:
            for larger in plans:
                if all(larger["sessions"][g] >= smaller["sessions"][g]
                       for g in self.minimum):
                    self.assertGreaterEqual(
                        larger["reliability_lower"],
                        smaller["reliability_lower"] - 1e-12)


if __name__ == "__main__":
    unittest.main()
