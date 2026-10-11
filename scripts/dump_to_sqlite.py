#!/usr/bin/env python3
"""Export and synchronize DuckDB data to an SQLite database.

Dumps all Annex B tables (orders, plants, transport_duties, capacity_events)
and system audit tables (ingest_log, rejected_rows, runs, traces) from
data/allocation.duckdb into a standard SQLite database (data/allocation.sqlite).

Can be executed standalone or imported:
    python scripts/dump_to_sqlite.py --duckdb data/allocation.duckdb --sqlite data/allocation.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd

# SQLite DDL equivalent to DuckDB schemas
SQLITE_TABLE_DDL: Dict[str, str] = {
    "orders": """
        CREATE TABLE IF NOT EXISTS orders (
            order_id    TEXT NOT NULL PRIMARY KEY,
            product     TEXT NOT NULL,
            qty         INTEGER NOT NULL,
            destination TEXT NOT NULL,
            due         TEXT NOT NULL
        );
    """,
    "plants": """
        CREATE TABLE IF NOT EXISTS plants (
            plant               TEXT NOT NULL,
            product             TEXT NOT NULL,
            weekly_capacity     INTEGER NOT NULL,
            unit_cost           REAL NOT NULL,
            energy_kwh_per_unit REAL NOT NULL,
            carbon_kg_per_kwh   REAL NOT NULL,
            PRIMARY KEY (plant, product)
        );
    """,
    "transport_duties": """
        CREATE TABLE IF NOT EXISTS transport_duties (
            plant                   TEXT NOT NULL,
            destination             TEXT NOT NULL,
            transport_cost_per_unit REAL NOT NULL,
            transit_days            INTEGER NOT NULL,
            duty_pct                REAL NOT NULL,
            carbon_kg_per_unit      REAL NOT NULL,
            PRIMARY KEY (plant, destination)
        );
    """,
    "capacity_events": """
        CREATE TABLE IF NOT EXISTS capacity_events (
            event_id          TEXT NOT NULL PRIMARY KEY,
            plant             TEXT NOT NULL,
            from_date         TEXT NOT NULL,
            to_date           TEXT NOT NULL,
            capacity_loss_pct REAL NOT NULL
        );
    """,
    "ingest_log": """
        CREATE TABLE IF NOT EXISTS ingest_log (
            ingest_id     TEXT PRIMARY KEY,
            table_name    TEXT NOT NULL,
            source        TEXT,
            mode          TEXT NOT NULL,
            rows_loaded   INTEGER NOT NULL,
            rows_rejected INTEGER NOT NULL,
            status        TEXT NOT NULL,
            loaded_at     TEXT NOT NULL
        );
    """,
    "rejected_rows": """
        CREATE TABLE IF NOT EXISTS rejected_rows (
            ingest_id  TEXT NOT NULL,
            table_name TEXT NOT NULL,
            row_number INTEGER NOT NULL,
            reason     TEXT NOT NULL,
            raw        TEXT
        );
    """,
    "runs": """
        CREATE TABLE IF NOT EXISTS runs (
            run_id      TEXT PRIMARY KEY,
            request     TEXT,
            answer_type TEXT,
            response    TEXT,
            latency_ms  REAL,
            created_at  TEXT NOT NULL
        );
    """,
    "traces": """
        CREATE TABLE IF NOT EXISTS traces (
            run_id     TEXT PRIMARY KEY,
            request    TEXT,
            trace      TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """,
}


def dump_duckdb_to_sqlite(
    duckdb_path: str = "data/allocation.duckdb",
    sqlite_path: str = "data/allocation.sqlite",
    verbose: bool = True,
) -> Dict[str, int]:
    """Extract all tables from DuckDB and insert into SQLite with schema creation."""
    d_path = Path(duckdb_path)
    if not d_path.exists():
        raise FileNotFoundError(f"DuckDB database not found at {duckdb_path}")

    s_path = Path(sqlite_path)
    s_path.parent.mkdir(parents=True, exist_ok=True)

    duck_con = duckdb.connect(str(d_path), read_only=True)
    sqlite_con = sqlite3.connect(str(s_path))
    sqlite_con.execute("PRAGMA journal_mode = WAL;")

    transferred_counts: Dict[str, int] = {}

    try:
        # Create SQLite schema
        cursor = sqlite_con.cursor()
        for table, ddl in SQLITE_TABLE_DDL.items():
            cursor.execute(ddl)
        sqlite_con.commit()

        # Discover all available tables in DuckDB
        duck_tables = [r[0] for r in duck_con.execute("SHOW TABLES").fetchall()]

        for table in duck_tables:
            # Read whole table into DataFrame
            df = duck_con.execute(f"SELECT * FROM {table}").df()
            
            # Format JSON / timestamp columns for SQLite storage
            for col in df.columns:
                if df[col].dtype == "object":
                    # Convert dicts/lists to JSON strings
                    df[col] = df[col].apply(lambda x: json.dumps(x, default=str) if isinstance(x, (dict, list)) else (str(x) if pd.notna(x) else None))
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

            # Replace contents in SQLite
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            df.to_sql(table, sqlite_con, if_exists="replace", index=False)
            sqlite_con.commit()

            count = len(df)
            transferred_counts[table] = count
            if verbose:
                print(f"  [EXPORTED] Table '{table}': {count} rows -> SQLite")

    finally:
        duck_con.close()
        sqlite_con.close()

    return transferred_counts


def verify_sqlite_dump(sqlite_path: str = "data/allocation.sqlite") -> Dict[str, int]:
    """Verify table counts in generated SQLite database."""
    s_path = Path(sqlite_path)
    if not s_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {sqlite_path}")

    con = sqlite3.connect(str(s_path))
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [r[0] for r in cursor.fetchall()]
    counts = {}
    for t in tables:
        cnt = cursor.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        counts[t] = cnt
    con.close()
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export DuckDB database to SQLite")
    parser.add_argument("--duckdb", default="data/allocation.duckdb", help="Source DuckDB database path")
    parser.add_argument("--sqlite", default="data/allocation.sqlite", help="Target SQLite database path")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing SQLite database")
    args = parser.parse_args(argv)

    if args.verify_only:
        print(f"Verifying SQLite database at {args.sqlite}...")
        counts = verify_sqlite_dump(args.sqlite)
        for t, n in counts.items():
            print(f"  {t}: {n} rows")
        return 0

    print(f"Exporting from DuckDB ({args.duckdb}) to SQLite ({args.sqlite})...")
    counts = dump_duckdb_to_sqlite(args.duckdb, args.sqlite, verbose=True)
    print(f"\nSuccessfully migrated {len(counts)} tables to SQLite: {args.sqlite}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
