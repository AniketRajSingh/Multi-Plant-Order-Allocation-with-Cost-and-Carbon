"""DuckDB persistence for the Annex B tables, ingest audit, run results and traces.

Design notes:
- One DuckDB file (DUCKDB_PATH, default data/allocation.duckdb).
- Connections are short-lived (opened per operation). DuckDB allows only one
  writer process per file, so holding a connection open in the API would block
  the CLI loader (scripts/load_data.py) while the API is running.
- Every CSV row is validated against its Annex B Pydantic row model. Invalid
  rows are rejected with a reason and recorded in `rejected_rows`; they never
  reach the data tables.
- Extra CSV columns are kept (Annex B allows them) and added to the table on demand.
"""

from __future__ import annotations

import io
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import duckdb
import pandas as pd
from pydantic import ValidationError

from app.api.schemas import ROW_MODELS


DEFAULT_DB_PATH = "data/allocation.duckdb"

# Base DDL for the Annex B tables. Extra columns are appended at ingest time.
TABLE_DDL: Dict[str, str] = {
    "orders": """
        order_id    VARCHAR NOT NULL,
        product     VARCHAR NOT NULL,
        qty         INTEGER NOT NULL,
        destination VARCHAR NOT NULL,
        due         DATE    NOT NULL
    """,
    "plants": """
        plant               VARCHAR NOT NULL,
        product             VARCHAR NOT NULL,
        weekly_capacity     INTEGER NOT NULL,
        unit_cost           DOUBLE  NOT NULL,
        energy_kwh_per_unit DOUBLE  NOT NULL,
        carbon_kg_per_kwh   DOUBLE  NOT NULL
    """,
    "transport_duties": """
        plant                   VARCHAR NOT NULL,
        destination             VARCHAR NOT NULL,
        transport_cost_per_unit DOUBLE  NOT NULL,
        transit_days            INTEGER NOT NULL,
        duty_pct                DOUBLE  NOT NULL,
        carbon_kg_per_unit      DOUBLE  NOT NULL
    """,
    "capacity_events": """
        event_id          VARCHAR NOT NULL,
        plant             VARCHAR NOT NULL,
        from_date         DATE    NOT NULL,
        to_date           DATE    NOT NULL,
        capacity_loss_pct DOUBLE  NOT NULL
    """,
}

# Natural key per table, used for de-duplication, append and upsert.
# Optional key columns join the key only when the file has them
# (e.g. a `week` column when carbon intensity varies by week).
TABLE_KEYS: Dict[str, List[str]] = {
    "orders": ["order_id"],
    "plants": ["plant", "product"],
    "transport_duties": ["plant", "destination"],
    "capacity_events": ["event_id"],
}
OPTIONAL_KEY_COLUMNS: Dict[str, List[str]] = {
    "plants": ["week", "week_start"],
    "transport_duties": ["product"],
}

# Tolerate common header variants so a late "schema change" drop still loads.
COLUMN_ALIASES: Dict[str, str] = {
    "quantity": "qty",
    "due_date": "due",
    "dest": "destination",
    "capacity": "weekly_capacity",
    "capacity_per_week": "weekly_capacity",
    "cost_per_unit": "unit_cost",
    "kwh_per_unit": "energy_kwh_per_unit",
    "carbon_intensity": "carbon_kg_per_kwh",
    "transport_cost": "transport_cost_per_unit",
    "transit_time_days": "transit_days",
    "duty": "duty_pct",
    "import_duty_pct": "duty_pct",
    "transport_carbon_kg_per_unit": "carbon_kg_per_unit",
    "start_date": "from_date",
    "end_date": "to_date",
    "loss_pct": "capacity_loss_pct",
}

SYSTEM_DDL = """
CREATE TABLE IF NOT EXISTS ingest_log (
    ingest_id     VARCHAR PRIMARY KEY,
    table_name    VARCHAR NOT NULL,
    source        VARCHAR,
    mode          VARCHAR NOT NULL,
    rows_loaded   INTEGER NOT NULL,
    rows_rejected INTEGER NOT NULL,
    status        VARCHAR NOT NULL,
    loaded_at     TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS rejected_rows (
    ingest_id  VARCHAR NOT NULL,
    table_name VARCHAR NOT NULL,
    row_number INTEGER NOT NULL,
    reason     VARCHAR NOT NULL,
    raw        JSON
);
CREATE TABLE IF NOT EXISTS runs (
    run_id      VARCHAR PRIMARY KEY,
    request     VARCHAR,
    answer_type VARCHAR,
    response    JSON,
    latency_ms  DOUBLE,
    created_at  TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    run_id     VARCHAR PRIMARY KEY,
    request    VARCHAR,
    trace      JSON NOT NULL,
    created_at TIMESTAMP NOT NULL
);
"""

MAX_REJECTED_RETURNED = 50


@dataclass
class IngestResult:
    table: str
    mode: str
    rows_loaded: int = 0
    rows_rejected: int = 0
    total_rows_in_table: int = 0
    errors: List[str] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.errors or (self.rows_loaded == 0 and self.rows_rejected > 0):
            return "failed"
        return "partial" if self.rows_rejected else "loaded"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "mode": self.mode,
            "rows_loaded": self.rows_loaded,
            "rows_rejected": self.rows_rejected,
            "status": self.status,
            "total_rows_in_table": self.total_rows_in_table,
            "errors": self.errors,
            "rejected": self.rejected[:MAX_REJECTED_RETURNED],
        }


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    severity: str = "error"  # "error" fails validation, "warning" does not

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.name, "passed": self.passed, "severity": self.severity, "detail": self.detail}


class Database:
    """Thin DuckDB wrapper. Safe to share across FastAPI worker threads."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv("DUCKDB_PATH", DEFAULT_DB_PATH)
        self._lock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # An in-memory database would vanish between short-lived connections,
        # so keep one connection open for that case (used by tests).
        self._memory_con = duckdb.connect(":memory:") if self.path == ":memory:" else None

    @contextmanager
    def connect(self, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
        with self._lock:
            if self._memory_con is not None:
                yield self._memory_con
                return
            con = duckdb.connect(self.path, read_only=read_only)
            try:
                yield con
            finally:
                con.close()

    # ------------------------------------------------------------------ schema

    def init_schema(self) -> None:
        """Create every table if missing. Idempotent."""
        with self.connect() as con:
            for table, ddl in TABLE_DDL.items():
                con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({ddl})")
            con.execute(SYSTEM_DDL)

    def table_counts(self) -> Dict[str, int]:
        with self.connect() as con:
            return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLE_DDL}

    def ping(self) -> bool:
        with self.connect() as con:
            return con.execute("SELECT 1").fetchone()[0] == 1

    # ------------------------------------------------------------------ ingest

    def ingest_csv(
        self,
        table: str,
        source: Union[str, Path, bytes],
        mode: str = "replace",
        source_name: Optional[str] = None,
    ) -> IngestResult:
        """Validate and load one Annex B CSV (a path or raw bytes) into `table`."""
        result = IngestResult(table=table, mode=mode)
        if table not in TABLE_DDL:
            result.errors.append(f"unknown table '{table}'; expected one of {sorted(TABLE_DDL)}")
            return result
        if mode not in ("replace", "append", "upsert"):
            result.errors.append(f"unknown mode '{mode}'; expected replace, append or upsert")
            return result

        try:
            buffer = io.BytesIO(source) if isinstance(source, bytes) else source
            raw = pd.read_csv(buffer, dtype=str, keep_default_na=False, skipinitialspace=True)
        except Exception as exc:  # malformed CSV, wrong encoding, empty file
            result.errors.append(f"could not parse CSV: {exc}")
            return result

        raw.columns = [_normalise_column(c) for c in raw.columns]
        if raw.columns.duplicated().any():
            dupes = sorted(set(raw.columns[raw.columns.duplicated()]))
            result.errors.append(f"duplicate columns after normalising headers: {dupes}")
            return result

        model = ROW_MODELS[table]
        missing = [c for c in model.model_fields if c not in raw.columns]
        if missing:
            result.rows_rejected = len(raw)
            result.errors.append(f"missing required columns {missing}; found {list(raw.columns)}")
            return result

        valid_rows, rejected = _validate_rows(raw, model)
        key_cols = TABLE_KEYS[table] + [c for c in OPTIONAL_KEY_COLUMNS.get(table, []) if c in raw.columns]
        valid_rows, dup_rejects = _drop_duplicate_keys(valid_rows, key_cols)
        rejected.extend(dup_rejects)

        df = pd.DataFrame([row for _, row in valid_rows])
        if not df.empty:
            extras = [c for c in df.columns if c not in model.model_fields]
            for col in extras:
                df[col] = _infer_extra_column(df[col])

        with self.connect() as con:
            con.execute("BEGIN TRANSACTION")
            try:
                if mode == "append" and not df.empty:
                    df, existing_rejects = self._drop_existing_keys(con, table, df, valid_rows, key_cols)
                    rejected.extend(existing_rejects)
                # A file with no valid rows never touches the table, so a bad
                # upload in replace mode cannot wipe the data already loaded.
                if not df.empty:
                    if mode == "replace":
                        con.execute(f"DROP TABLE IF EXISTS {table}")
                        con.execute(f"CREATE TABLE {table} ({TABLE_DDL[table]})")
                    self._add_missing_columns(con, table, df)
                    con.register("incoming", df)
                    if mode == "upsert":
                        join = " AND ".join(f"{table}.{k} = incoming.{k}" for k in key_cols)
                        con.execute(f"DELETE FROM {table} USING incoming WHERE {join}")
                    con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM incoming")
                    con.unregister("incoming")

                result.rows_loaded = len(df)
                result.rows_rejected = len(rejected)
                result.rejected = [
                    {"row_number": n, "reason": r} for n, r, _ in sorted(rejected, key=lambda x: x[0])
                ]
                result.total_rows_in_table = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                self._log_ingest(con, result, source_name or _source_label(source), rejected)
                con.execute("COMMIT")
            except Exception as exc:
                con.execute("ROLLBACK")
                result.rows_loaded = 0
                result.errors.append(f"database write failed: {exc}")
        return result

    @staticmethod
    def _add_missing_columns(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
        existing = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
        for col in df.columns:
            if col not in existing:
                sql_type = "DOUBLE" if pd.api.types.is_numeric_dtype(df[col]) else "VARCHAR"
                con.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" {sql_type}')

    @staticmethod
    def _drop_existing_keys(
        con: duckdb.DuckDBPyConnection,
        table: str,
        df: pd.DataFrame,
        valid_rows: List[Tuple[int, Dict[str, Any]]],
        key_cols: List[str],
    ) -> Tuple[pd.DataFrame, List[Tuple[int, str, Dict[str, Any]]]]:
        cols = ", ".join(key_cols)
        existing = {tuple(str(v) for v in r) for r in con.execute(f"SELECT {cols} FROM {table}").fetchall()}
        keep, rejects = [], []
        for i, (row_number, row) in enumerate(valid_rows):
            key = tuple(str(row[k]) for k in key_cols)
            if key in existing:
                rejects.append((row_number, f"key {dict(zip(key_cols, key))} already in table (use mode=upsert)", row))
            else:
                keep.append(i)
        return df.iloc[keep].reset_index(drop=True), rejects

    def _log_ingest(
        self,
        con: duckdb.DuckDBPyConnection,
        result: IngestResult,
        source: str,
        rejected: List[Tuple[int, str, Dict[str, Any]]],
    ) -> None:
        ingest_id = f"ing-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        con.execute(
            "INSERT INTO ingest_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [ingest_id, result.table, source, result.mode, result.rows_loaded,
             result.rows_rejected, result.status, _now()],
        )
        if rejected:
            con.executemany(
                "INSERT INTO rejected_rows VALUES (?, ?, ?, ?, ?)",
                [[ingest_id, result.table, n, reason, json.dumps(raw, default=str)] for n, reason, raw in rejected],
            )

    # ------------------------------------------------------------------ reads

    def get_table(self, table: str) -> pd.DataFrame:
        """Return a whole Annex B table as a DataFrame (what the solver consumes)."""
        if table not in TABLE_DDL:
            raise ValueError(f"unknown table '{table}'")
        with self.connect() as con:
            return con.execute(f"SELECT * FROM {table}").df()

    def load_dataset(self) -> Dict[str, pd.DataFrame]:
        return {t: self.get_table(t) for t in TABLE_DDL}

    # ------------------------------------------------------------------ runs and traces

    def save_run(self, run_id: str, request: str, answer_type: str, response: Dict[str, Any], latency_ms: float) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                [run_id, request, answer_type, json.dumps(response, default=str), latency_ms, _now()],
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as con:
            row = con.execute("SELECT response FROM runs WHERE run_id = ?", [run_id]).fetchone()
        return json.loads(row[0]) if row else None

    def save_trace(self, trace: Dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?)",
                [trace["run_id"], trace.get("request", ""), json.dumps(trace, default=str), _now()],
            )

    def get_trace(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as con:
            row = con.execute("SELECT trace FROM traces WHERE run_id = ?", [run_id]).fetchone()
        return json.loads(row[0]) if row else None

    # ------------------------------------------------------------------ Section 4.5 checks

    def run_validation_checks(
        self,
        planning_start: Optional[date] = None,
        expected_counts: Optional[Dict[str, int]] = None,
    ) -> List[CheckResult]:
        """Cross-table checks from Section 4.5. Row-level checks already ran at ingest."""
        planning_start = planning_start or _planning_start_from_env()
        checks: List[CheckResult] = []
        with self.connect() as con:
            q = lambda sql: con.execute(sql).fetchall()  # noqa: E731

            counts = {t: q(f"SELECT COUNT(*) FROM {t}")[0][0] for t in TABLE_DDL}
            for t in ("orders", "plants", "transport_duties"):
                checks.append(CheckResult(f"{t} loaded", counts[t] > 0, f"{counts[t]} rows"))

            no_plant = [r[0] for r in q(
                "SELECT DISTINCT o.product FROM orders o "
                "WHERE o.product NOT IN (SELECT product FROM plants) ORDER BY 1"
            )]
            checks.append(CheckResult(
                "every order's product can be made in at least one plant",
                not no_plant,
                f"products with no plant: {no_plant}" if no_plant else "all products covered",
            ))

            single = q(
                "SELECT product, MIN(plant) FROM plants GROUP BY product "
                "HAVING COUNT(DISTINCT plant) = 1 ORDER BY 1"
            )
            checks.append(CheckResult(
                "single-plant products (edge case, informational)", True,
                f"{len(single)} product(s) made by one plant only: {[f'{p}->{pl}' for p, pl in single]}",
                severity="warning",
            ))

            missing_lanes = q(
                "WITH p AS (SELECT DISTINCT plant FROM plants), "
                "d AS (SELECT DISTINCT destination FROM orders UNION SELECT DISTINCT destination FROM transport_duties) "
                "SELECT p.plant, d.destination FROM p CROSS JOIN d "
                "WHERE NOT EXISTS (SELECT 1 FROM transport_duties t "
                "                  WHERE t.plant = p.plant AND t.destination = d.destination) ORDER BY 1, 2"
            )
            checks.append(CheckResult(
                "every plant and destination pair has a transport row",
                not missing_lanes,
                f"{len(missing_lanes)} missing: {[f'{p}->{d}' for p, d in missing_lanes[:20]]}"
                if missing_lanes else "all lanes present",
            ))

            bad_cap = q("SELECT COUNT(*) FROM plants WHERE weekly_capacity <= 0")[0][0]
            checks.append(CheckResult(
                "capacity is positive (unit: units per week)", bad_cap == 0,
                f"{bad_cap} non-positive capacity rows" if bad_cap else "all capacities > 0 units/week",
            ))

            early = q(f"SELECT order_id, due FROM orders WHERE due <= DATE '{planning_start.isoformat()}' ORDER BY due")
            checks.append(CheckResult(
                f"due dates are after the planning start ({planning_start.isoformat()})",
                not early,
                f"{len(early)} orders due on/before start: {[r[0] for r in early[:20]]}" if early else "all due dates later",
            ))

            # kg CO2e/kWh: real grids sit roughly 0.0-1.2. Values far above suggest g/kWh was supplied.
            ci_min, ci_max = q("SELECT MIN(carbon_kg_per_kwh), MAX(carbon_kg_per_kwh) FROM plants")[0]
            ci_ok = ci_max is None or ci_max <= 2.0
            checks.append(CheckResult(
                "carbon intensity in stated unit (kg CO2e per kWh, expected 0-2)", ci_ok,
                f"range {ci_min}-{ci_max} kg/kWh" + ("" if ci_ok else " - looks like g/kWh, not kg/kWh"),
            ))

            unknown_event_plants = [r[0] for r in q(
                "SELECT DISTINCT plant FROM capacity_events WHERE plant NOT IN (SELECT plant FROM plants)"
            )]
            checks.append(CheckResult(
                "capacity events refer to known plants", not unknown_event_plants,
                f"unknown plants: {unknown_event_plants}" if unknown_event_plants else f"{counts['capacity_events']} events ok",
            ))

            fraction_like = q(
                "SELECT COUNT(*) FROM capacity_events WHERE capacity_loss_pct > 0 AND capacity_loss_pct <= 1"
            )[0][0]
            if fraction_like:
                checks.append(CheckResult(
                    "capacity_loss_pct is a percentage (0-100)", True,
                    f"{fraction_like} event(s) have loss <= 1; confirm these are percent, not fractions",
                    severity="warning",
                ))

            for table, expected in (expected_counts or {}).items():
                actual = counts.get(table)
                checks.append(CheckResult(
                    f"row count matches data card: {table}", actual == expected,
                    f"expected {expected}, found {actual}",
                ))
        return checks


# ============================================================================
# helpers
# ============================================================================

def _normalise_column(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_").lstrip("﻿")
    return COLUMN_ALIASES.get(key, key)


def _validate_rows(raw: pd.DataFrame, model) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, str, Dict[str, Any]]]]:
    valid: List[Tuple[int, Dict[str, Any]]] = []
    rejected: List[Tuple[int, str, Dict[str, Any]]] = []
    for idx, record in enumerate(raw.to_dict(orient="records"), start=1):
        cleaned = {k: (None if isinstance(v, str) and v.strip() == "" else v) for k, v in record.items()}
        try:
            valid.append((idx, model.model_validate(cleaned).model_dump()))
        except ValidationError as exc:
            reason = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'row'}: {e['msg']}" for e in exc.errors())
            rejected.append((idx, reason, record))
    return valid, rejected


def _drop_duplicate_keys(
    rows: List[Tuple[int, Dict[str, Any]]], key_cols: List[str]
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, str, Dict[str, Any]]]]:
    seen: Dict[tuple, int] = {}
    keep, rejects = [], []
    for row_number, row in rows:
        key = tuple(row.get(k) for k in key_cols)
        if key in seen:
            rejects.append((row_number, f"duplicate key {dict(zip(key_cols, key))} (first seen on row {seen[key]})", row))
        else:
            seen[key] = row_number
            keep.append((row_number, row))
    return keep, rejects


def _infer_extra_column(col: pd.Series) -> pd.Series:
    """Extra columns arrive as strings; make them numeric when every value parses."""
    numeric = pd.to_numeric(col, errors="coerce")
    if numeric.notna().sum() == col.notna().sum():
        return numeric
    return col


def _source_label(source: Union[str, Path, bytes]) -> str:
    return "upload" if isinstance(source, bytes) else str(source)


def _planning_start_from_env() -> date:
    value = os.getenv("PLANNING_START")
    return date.fromisoformat(value) if value else date.today()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


_db: Optional[Database] = None
_db_lock = threading.Lock()


def set_db(db: Optional[Database]) -> None:
    """Swap the process-wide Database (tests, or pointing the API at another file)."""
    global _db
    with _db_lock:
        _db = db


def get_db() -> Database:
    """Process-wide Database (also the FastAPI dependency)."""
    global _db
    with _db_lock:
        if _db is None:
            _db = Database()
            _db.init_schema()
        return _db
