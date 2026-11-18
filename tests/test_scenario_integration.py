import json
import tempfile
import unittest
from pathlib import Path

from learning_mobile_capacity.capacity import decide, load_resources
from learning_mobile_capacity.forecast import forecast_demand
from learning_mobile_capacity.records import read_records
from learning_mobile_capacity.scenario_report import build_report, render
from learning_mobile_capacity.storage import replace_records

DATA = Path(__file__).parent.parent / "data"


class ScenarioIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_records(DATA / "demand_history.csv")
        self.resources = load_resources(DATA / "resources.json")
        self.commitments = json.loads(
            (DATA / "commitments.json").read_text(encoding="utf-8"))

    def test_cutoff_excludes_july_actuals_from_forecast(self):
        forecast = forecast_demand(self.rows, self.commitments["as_of"])
        periods = set()
        for marginal in forecast["communities"].values():
            periods.update(marginal["periods"])
        self.assertNotIn("2026-07", periods)

    def test_full_report_renders_and_carries_evidence_sections(self):
        forecast, decision = build_report(self.rows, self.resources,
                                          self.commitments)
        text = render(forecast, decision)
        for heading in ("联合需求与运力决策报告", "联合情景的证据基础",
                        "风险登记", "无法作出的承诺与可调整范围"):
            self.assertIn(heading, text)
        # 必须如实说明相依性未知，而不是假设独立
        self.assertIn("相关性未知", text)
        self.assertIn("Fréchet", text)
        self.assertEqual(decision["forecast_as_of"],
                         self.commitments["as_of"])

    def test_minimum_commitments_are_present_in_every_plan(self):
        forecast, decision = build_report(self.rows, self.resources,
                                          self.commitments)
        self.assertTrue(decision["plans"])
        for plan in decision["plans"]:
            for group, floor in self.commitments["minimum_sessions"].items():
                self.assertGreaterEqual(plan["sessions"][group], floor)

    def test_storage_roundtrip_keeps_legacy_pipeline_working(self):
        with tempfile.TemporaryDirectory() as directory:
            stored = replace_records(Path(directory) / "observations.db",
                                     self.rows)
        self.assertEqual(len(stored), len(self.rows))
        self.assertTrue(all(isinstance(row[2], int) for row in stored))


if __name__ == "__main__":
    unittest.main()
