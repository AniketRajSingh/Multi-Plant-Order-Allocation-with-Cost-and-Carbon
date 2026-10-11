"""Load Annex B CSV files into the SQLite store, then run the Section 4.5 validation checks.

Examples
--------
  # Create an empty database (schema only)
  python scripts/load_data.py --init-only

  # Load every Annex B file found in a folder (orders.csv, plants.csv, ...)
  python scripts/load_data.py --dir data/raw

  # Load individual files; append the late capacity-event drop
  python scripts/load_data.py --capacity-events drop/capacity_events.csv --mode append

  # Send files through a running API instead of writing the DB file directly
  python scripts/load_data.py --dir data/raw --api-url http://localhost:8000

  # Only run the validation checks on what is already loaded
  python scripts/load_data.py --check-only --planning-start 2026-10-12 --expect orders=150

Exit code is 1 if any file fails to load or any validation check (severity=error) fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.database import Database  # noqa: E402

# Load order matters only for readability of the output; tables are independent.
TABLES = ["plants", "transport_duties", "orders", "capacity_events"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, help="folder holding <table>.csv files")
    for table in TABLES:
        p.add_argument(f"--{table.replace('_', '-')}", dest=table, type=Path, help=f"path to {table}.csv")
    p.add_argument("--mode", choices=["replace", "append", "upsert"], default="replace",
                   help="replace the table (default), append new keys only, or upsert by key")
    p.add_argument("--db", help="SQLite database file (default: $DB_PATH or data/allocation.db)")
    p.add_argument("--api-url", help="POST files to <api-url>/ingest instead of writing the DB directly")
    p.add_argument("--planning-start", type=date.fromisoformat,
                   help="planning start date for the due-date check (default: $PLANNING_START or today)")
    p.add_argument("--expect", action="append", default=[], metavar="TABLE=N",
                   help="expected row count from the data card; repeatable")
    p.add_argument("--init-only", action="store_true", help="create the schema and exit")
    p.add_argument("--check-only", action="store_true", help="skip loading; run validation checks only")
    p.add_argument("--json", action="store_true", help="print a JSON report instead of text")
    return p.parse_args(argv)


def collect_files(args: argparse.Namespace) -> Tuple[Dict[str, Path], List[str]]:
    files: Dict[str, Path] = {}
    problems: List[str] = []
    if args.dir:
        if not args.dir.is_dir():
            problems.append(f"--dir {args.dir} is not a folder")
        else:
            for table in TABLES:
                candidate = args.dir / f"{table}.csv"
                if candidate.exists():
                    files[table] = candidate
    for table in TABLES:
        explicit = getattr(args, table)
        if explicit:
            if explicit.exists():
                files[table] = explicit
            else:
                problems.append(f"{table}: file not found: {explicit}")
    return files, problems


def parse_expected(pairs: List[str]) -> Dict[str, int]:
    out = {}
    for pair in pairs:
        table, _, n = pair.partition("=")
        if table not in TABLES or not n.isdigit():
            raise SystemExit(f"--expect must look like orders=150 with a known table, got '{pair}'")
        out[table] = int(n)
    return out


def ingest_via_api(api_url: str, table: str, path: Path, mode: str) -> dict:
    import httpx

    with path.open("rb") as fh:
        resp = httpx.post(
            f"{api_url.rstrip('/')}/ingest",
            files={"file": (path.name, fh, "text/csv")},
            data={"metadata": json.dumps({"table": table, "mode": mode})},
            timeout=120,
        )
    try:
        return resp.json()
    except ValueError:
        return {"table": table, "status": "failed", "rows_loaded": 0, "rows_rejected": 0,
                "errors": [f"HTTP {resp.status_code}: {resp.text[:300]}"]}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    expected = parse_expected(args.expect)
    db = Database(args.db) if args.db else Database()
    db.init_schema()

    if args.init_only:
        print(f"Initialised SQLite schema at {db.path}")
        print("Tables:", json.dumps(db.table_counts()))
        return 0

    report = {"database": db.path, "loads": [], "checks": []}
    failed = False

    if not args.check_only:
        files, problems = collect_files(args)
        for msg in problems:
            report["loads"].append({"status": "failed", "errors": [msg]})
            failed = True
        if not files and not problems:
            print("No files given. Use --dir or --orders/--plants/--transport-duties/--capacity-events.",
                  file=sys.stderr)
            return 2
        for table in TABLES:
            if table not in files:
                continue
            path = files[table]
            if args.api_url:
                res = ingest_via_api(args.api_url, table, path, args.mode)
            else:
                res = db.ingest_csv(table, path, mode=args.mode).to_dict()
            res["file"] = str(path)
            report["loads"].append(res)
            failed |= res.get("status") == "failed"

    # With --api-url the API owns the file, but SQLite WAL mode supports concurrent reads,
    # so reading it here for the checks still works when run on the same host.
    checks = db.run_validation_checks(planning_start=args.planning_start, expected_counts=expected)
    report["checks"] = [c.to_dict() for c in checks]
    failed |= any(not c.passed and c.severity == "error" for c in checks)
    report["result"] = "FAIL" if failed else "PASS"

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_text_report(report)
    return 1 if failed else 0


def print_text_report(report: dict) -> None:
    print(f"Database: {report['database']}")
    if report["loads"]:
        print("\nLoad results")
        print("-" * 72)
        for res in report["loads"]:
            name = res.get("table", "?")
            print(f"[{res.get('status', '?').upper():7}] {name:17} loaded={res.get('rows_loaded', 0):<5} "
                  f"rejected={res.get('rows_rejected', 0):<5} total={res.get('total_rows_in_table', '-')}  "
                  f"{res.get('file', '')}")
            for err in res.get("errors", []):
                print(f"          error: {err}")
            for rej in res.get("rejected", [])[:10]:
                print(f"          row {rej['row_number']}: {rej['reason']}")
            if res.get("rows_rejected", 0) > 10:
                print(f"          ... {res['rows_rejected'] - 10} more (see rejected_rows table)")

    print("\nValidation checks (Section 4.5)")
    print("-" * 72)
    for c in report["checks"]:
        mark = "PASS" if c["passed"] else ("WARN" if c["severity"] == "warning" else "FAIL")
        if c["severity"] == "warning" and c["passed"]:
            mark = "INFO"
        print(f"[{mark}] {c['check']}: {c['detail']}")
    print(f"\nRESULT: {report['result']}")


if __name__ == "__main__":
    sys.exit(main())
