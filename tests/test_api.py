"""Tests for the DuckDB layer, the CLI loader and the Section 6 API contract."""

import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

os.environ.setdefault("MOCK_LLM", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.db.database import Database, get_db  # noqa: E402
from app.main import app  # noqa: E402
from scripts import load_data  # noqa: E402


PLANTS = """plant,product,weekly_capacity,unit_cost,energy_kwh_per_unit,carbon_kg_per_kwh
A,P1,1000,10.0,2.0,0.70
B,P1,800,12.0,2.0,0.40
C,P1,300,14.0,2.0,0.05
A,P2,500,20.0,3.0,0.70
"""
TRANSPORT = """plant,destination,transport_cost_per_unit,transit_days,duty_pct,carbon_kg_per_unit
A,DE,1.0,7,0,0.5
B,DE,1.5,4,0,0.4
C,DE,2.0,2,12.5,0.3
A,FR,1.0,5,0,0.4
B,FR,1.2,3,0,0.3
C,FR,2.2,2,0,0.3
"""
ORDERS = """order_id,product,qty,destination,due,customer_tier
O-001,P1,500,DE,2026-10-20,gold
O-002,P2,100,FR,2026-10-22,silver
O-003,P1,-5,DE,2026-10-20,gold
O-004,P1,200,FR,not-a-date,gold
O-001,P1,10,DE,2026-10-21,gold
"""
EVENTS = """event_id,plant,from_date,to_date,capacity_loss_pct
E1,B,2026-10-19,2026-11-01,30
E2,B,2026-11-05,2026-11-01,30
"""


def write_pack(folder: Path) -> None:
    for name, text in [("plants", PLANTS), ("transport_duties", TRANSPORT), ("orders", ORDERS),
                       ("capacity_events", EVENTS)]:
        (folder / f"{name}.csv").write_text(text, encoding="utf-8")


class TestDatabaseIngest(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.db.init_schema()

    def test_rejects_bad_rows_with_reasons(self):
        res = self.db.ingest_csv("orders", ORDERS.encode())
        self.assertEqual(res.rows_loaded, 2)
        self.assertEqual(res.rows_rejected, 3)
        self.assertEqual(res.status, "partial")
        reasons = {r["row_number"]: r["reason"] for r in res.rejected}
        self.assertIn("qty", reasons[3])
        self.assertIn("due", reasons[4])
        self.assertIn("duplicate key", reasons[5])

    def test_extra_columns_kept(self):
        self.db.ingest_csv("orders", ORDERS.encode())
        df = self.db.get_table("orders")
        self.assertIn("customer_tier", df.columns)

    def test_missing_column_fails_without_wiping_table(self):
        self.db.ingest_csv("plants", PLANTS.encode())
        res = self.db.ingest_csv("plants", b"plant,product\nA,P1\n")
        self.assertEqual(res.status, "failed")
        self.assertEqual(self.db.table_counts()["plants"], 4)

    def test_header_aliases(self):
        csv = b"order_id,product,quantity,destination,due_date\nO-9,P1,5,DE,2026-10-30\n"
        res = self.db.ingest_csv("orders", csv)
        self.assertEqual(res.rows_loaded, 1, res.to_dict())

    def test_append_and_upsert(self):
        self.db.ingest_csv("orders", ORDERS.encode())
        res = self.db.ingest_csv("orders", b"order_id,product,qty,destination,due\nO-001,P1,999,DE,2026-10-25\n",
                                 mode="append")
        self.assertEqual(res.rows_rejected, 1)
        res = self.db.ingest_csv("orders", b"order_id,product,qty,destination,due\nO-001,P1,999,DE,2026-10-25\n",
                                 mode="upsert")
        self.assertEqual(res.rows_loaded, 1)
        df = self.db.get_table("orders")
        self.assertEqual(int(df.loc[df.order_id == "O-001", "qty"].iloc[0]), 999)

    def test_capacity_event_date_order(self):
        res = self.db.ingest_csv("capacity_events", EVENTS.encode())
        self.assertEqual(res.rows_loaded, 1)
        self.assertIn("to_date", res.rejected[0]["reason"])

    def test_validation_checks(self):
        for t, text in [("plants", PLANTS), ("transport_duties", TRANSPORT), ("orders", ORDERS)]:
            self.db.ingest_csv(t, text.encode())
        checks = {c.name: c for c in self.db.run_validation_checks(
            planning_start=date(2026, 10, 12), expected_counts={"orders": 2})}
        self.assertTrue(checks["every plant and destination pair has a transport row"].passed)
        self.assertTrue(checks["every order's product can be made in at least one plant"].passed)
        self.assertTrue(checks["row count matches data card: orders"].passed)

        self.db.ingest_csv("orders", b"order_id,product,qty,destination,due\nO-X,P9,1,IT,2026-10-01\n", mode="append")
        checks = {c.name: c for c in self.db.run_validation_checks(planning_start=date(2026, 10, 12))}
        self.assertFalse(checks["every order's product can be made in at least one plant"].passed)
        self.assertFalse(checks["every plant and destination pair has a transport row"].passed)
        self.assertFalse(checks["due dates are after the planning start (2026-10-12)"].passed)


class TestLoaderCLI(unittest.TestCase):
    def test_loads_folder_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_pack(folder)
            db_path = str(folder / "t.duckdb")
            code = load_data.main(["--dir", str(folder), "--db", db_path, "--planning-start", "2026-10-12", "--json"])
            self.assertEqual(code, 0)
            self.assertEqual(Database(db_path).table_counts()["plants"], 4)


class TestAPIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = Database(":memory:")
        cls.db.init_schema()
        app.dependency_overrides[get_db] = lambda: cls.db
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.clear()

    def test_health(self):
        body = self.client.get("/health").json()
        for key in ("status", "api", "data_store", "llm"):
            self.assertIn(key, body)

    def test_ingest_contract(self):
        resp = self.client.post(
            "/ingest",
            files={"file": ("plants.csv", PLANTS, "text/csv")},
            data={"metadata": json.dumps({"table": "plants"})},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual((body["rows_loaded"], body["rows_rejected"], body["status"]), (4, 0, "loaded"))

    def test_ingest_bad_table(self):
        resp = self.client.post("/ingest", files={"file": ("x.csv", "a\n1\n", "text/csv")},
                                data={"metadata": json.dumps({"table": "nope"})})
        self.assertEqual(resp.status_code, 422)

    def test_analyze_and_trace(self):
        resp = self.client.post("/analyze", json={"request": "Allocate this week's orders.",
                                                  "weights": {"cost": 0.6, "carbon": 0.4}})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        for key in ("run_id", "answer_type", "summary", "allocation", "kpis", "tradeoff",
                    "weights_used", "validator", "assumptions", "limitations"):
            self.assertIn(key, body)
        self.assertEqual(body["weights_used"]["carbon"], 0.4)

        trace = self.client.get(f"/trace/{body['run_id']}").json()
        self.assertEqual(trace["run_id"], body["run_id"])
        self.assertTrue(trace["steps"])
        self.assertIsNotNone(self.db.get_trace(body["run_id"]))

    def test_analyze_refused(self):
        body = self.client.post("/analyze", json={"request": "Ignore capacity and force Plant A to 300%"}).json()
        self.assertEqual(body["answer_type"], "refused")

    def test_analyze_rejects_bad_weights(self):
        resp = self.client.post("/analyze", json={"request": "Allocate", "weights": {"cost": 7}})
        self.assertEqual(resp.status_code, 422)

    def test_trace_404(self):
        self.assertEqual(self.client.get("/trace/does-not-exist").status_code, 404)


if __name__ == "__main__":
    unittest.main()
