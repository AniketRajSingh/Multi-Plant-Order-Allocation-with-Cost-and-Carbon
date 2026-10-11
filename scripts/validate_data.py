#!/usr/bin/env python3
"""Standalone CLI dataset validator for Annex B CSV files.

This script performs pre-ingestion checks on 4 CSV files:
  - orders.csv
  - plants.csv
  - transport_duties.csv
  - capacity_events.csv (optional)

It uses pure Python + pandas (no backend / database imports) so that data quality
can be verified before running ingestion or database loads.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Mapping of common column header variants / aliases to canonical Annex B column names
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

# Required columns for each Annex B table
REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "orders": ["order_id", "product", "qty", "destination", "due"],
    "plants": [
        "plant",
        "product",
        "weekly_capacity",
        "unit_cost",
        "energy_kwh_per_unit",
        "carbon_kg_per_kwh",
    ],
    "transport_duties": [
        "plant",
        "destination",
        "transport_cost_per_unit",
        "transit_days",
        "duty_pct",
        "carbon_kg_per_unit",
    ],
    "capacity_events": [
        "event_id",
        "plant",
        "from_date",
        "to_date",
        "capacity_loss_pct",
    ],
}


def apply_column_aliases(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Standardize column names by applying known aliases and stripping whitespace."""
    # Clean whitespace and lowercase for reliable alias matching
    cleaned_columns = {}
    for col in df.columns:
        col_str = str(col).strip()
        canonical = COLUMN_ALIASES.get(col_str.lower(), col_str)
        cleaned_columns[col] = canonical
    return df.rename(columns=cleaned_columns)


def check_required_columns(df: pd.DataFrame, table_name: str) -> Tuple[bool, str]:
    """Check if all required columns for the table are present."""
    required = REQUIRED_COLUMNS.get(table_name, [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"
    return True, "Required columns present"


def check_no_missing(df: pd.DataFrame, columns: List[str], label: str) -> Tuple[bool, str]:
    """Check that specified columns do not contain NaN, null, or empty string values."""
    if df.empty:
        return True, f"No missing values in {label} (0 rows)"

    missing_info = []
    for col in columns:
        if col not in df.columns:
            continue
        # Check for NaN / None
        null_mask = df[col].isna()
        # For object/string columns, also check for whitespace-only strings
        if df[col].dtype == object or df[col].dtype == "string":
            empty_str_mask = df[col].astype(str).str.strip() == ""
            combined_mask = null_mask | empty_str_mask
        else:
            combined_mask = null_mask

        missing_count = combined_mask.sum()
        if missing_count > 0:
            missing_info.append(f"{col} ({missing_count} missing)")

    if missing_info:
        return False, f"Missing values found in {label}: {', '.join(missing_info)}"
    return True, f"No missing values in {label}"


def check_duplicates(df: pd.DataFrame, key_columns: List[str], label: str) -> Tuple[bool, str]:
    """Check for duplicate rows across primary key columns."""
    if df.empty:
        return True, f"No duplicate {label} found (0 rows)"

    # Filter to columns that actually exist
    keys = [c for c in key_columns if c in df.columns]
    if len(keys) != len(key_columns):
        return False, f"Cannot check duplicates: missing key column(s) in {key_columns}"

    dup_mask = df.duplicated(subset=keys, keep=False)
    if dup_mask.any():
        dup_rows = df[dup_mask]
        # Group by keys to show examples with 1-indexed row numbers (like in a spreadsheet / CSV editor)
        first_group = dup_rows.groupby(keys).groups
        sample_key = next(iter(first_group))
        sample_indices = [idx + 2 for idx in first_group[sample_key][:3]]  # +2 accounts for header + 0-index
        key_str = sample_key if isinstance(sample_key, str) else ", ".join(map(str, sample_key))
        return False, f"Duplicate {label} found: {key_str} (rows {', '.join(map(str, sample_indices))})"

    return True, f"No duplicate {label} found"


def check_positive_integers(df: pd.DataFrame, column: str, label: str) -> Tuple[bool, str]:
    """Check that all values in a column are positive integers (> 0)."""
    if column not in df.columns:
        return False, f"Column '{column}' not found"
    if df.empty:
        return True, f"All {column} values are positive integers (0 rows)"

    invalid_indices = []
    for idx, val in df[column].dropna().items():
        # Check if integer or whole number float
        try:
            num = float(val)
            if not num.is_integer() or num <= 0:
                invalid_indices.append((idx + 2, val))
        except (ValueError, TypeError):
            invalid_indices.append((idx + 2, val))

    if invalid_indices:
        samples = [f"row {row}: {val}" for row, val in invalid_indices[:3]]
        return False, f"Invalid {label} (must be integer > 0): {', '.join(samples)}"
    return True, f"All {label} values are positive integers"


def check_non_negative(df: pd.DataFrame, column: str, label: str) -> Tuple[bool, str]:
    """Check that all values in a column are numeric and non-negative (>= 0)."""
    if column not in df.columns:
        return False, f"Column '{column}' not found"
    if df.empty:
        return True, f"All {label} values are >= 0 (0 rows)"

    invalid_indices = []
    for idx, val in df[column].dropna().items():
        try:
            num = float(val)
            if num < 0:
                invalid_indices.append((idx + 2, val))
        except (ValueError, TypeError):
            invalid_indices.append((idx + 2, val))

    if invalid_indices:
        samples = [f"row {row}: {val}" for row, val in invalid_indices[:3]]
        return False, f"Invalid {label} (must be >= 0): {', '.join(samples)}"
    return True, f"All {label} values are non-negative"


def check_range(
    df: pd.DataFrame, column: str, min_val: float, max_val: float, label: str
) -> Tuple[bool, str]:
    """Check that all numeric values in a column fall between min_val and max_val (inclusive)."""
    if column not in df.columns:
        return False, f"Column '{column}' not found"
    if df.empty:
        return True, f"All {label} values in range [{min_val}, {max_val}] (0 rows)"

    invalid_indices = []
    for idx, val in df[column].dropna().items():
        try:
            num = float(val)
            if num < min_val or num > max_val:
                invalid_indices.append((idx + 2, val))
        except (ValueError, TypeError):
            invalid_indices.append((idx + 2, val))

    if invalid_indices:
        samples = [f"row {row}: {val}" for row, val in invalid_indices[:3]]
        return False, f"Invalid {label} (must be between {min_val} and {max_val}): {', '.join(samples)}"
    return True, f"All {label} values are within [{min_val}, {max_val}]"


def check_valid_dates(df: pd.DataFrame, column: str, label: str) -> Tuple[bool, str]:
    """Check that all dates in a column follow the YYYY-MM-DD format."""
    if column not in df.columns:
        return False, f"Column '{column}' not found"
    if df.empty:
        return True, f"All {label} dates are valid (0 rows)"

    invalid_indices = []
    for idx, val in df[column].dropna().items():
        val_str = str(val).strip()
        try:
            # Parse strictly YYYY-MM-DD
            datetime.strptime(val_str, "%Y-%m-%d")
        except ValueError:
            invalid_indices.append((idx + 2, val))

    if invalid_indices:
        samples = [f"row {row}: '{val}'" for row, val in invalid_indices[:3]]
        return False, f"Invalid {label} date format (expected YYYY-MM-DD): {', '.join(samples)}"
    return True, f"All {label} are valid dates"


def check_date_order(df: pd.DataFrame, col_start: str, col_end: str, label: str) -> Tuple[bool, str]:
    """Check that to_date is greater than or equal to from_date."""
    if col_start not in df.columns or col_end not in df.columns:
        return False, f"Columns '{col_start}' or '{col_end}' missing for date order check"
    if df.empty:
        return True, f"Date order {col_end} >= {col_start} holds (0 rows)"

    invalid_rows = []
    for idx, row in df.iterrows():
        start_val = str(row[col_start]).strip()
        end_val = str(row[col_end]).strip()
        try:
            d_start = datetime.strptime(start_val, "%Y-%m-%d").date()
            d_end = datetime.strptime(end_val, "%Y-%m-%d").date()
            if d_end < d_start:
                invalid_rows.append((idx + 2, f"{end_val} < {start_val}"))
        except ValueError:
            # Date format issues are caught by check_valid_dates
            continue

    if invalid_rows:
        samples = [f"row {row}: {detail}" for row, detail in invalid_rows[:3]]
        return False, f"Date order violation in {label} ({col_end} < {col_start}): {', '.join(samples)}"
    return True, f"All date ranges in {label} are ordered ({col_end} >= {col_start})"


def check_product_eligibility(orders_df: pd.DataFrame, plants_df: pd.DataFrame) -> Tuple[bool, str]:
    """Check that every product requested in orders appears in at least one plant row."""
    if "product" not in orders_df.columns:
        return False, "Cannot verify product eligibility: orders.csv missing 'product' column"
    if "product" not in plants_df.columns:
        return False, "Cannot verify product eligibility: plants.csv missing 'product' column"

    order_products = set(orders_df["product"].dropna().astype(str).str.strip())
    plant_products = set(plants_df["product"].dropna().astype(str).str.strip())

    unsupported = order_products - plant_products
    if unsupported:
        return False, f"Unsupported products in orders (no plant makes them): {', '.join(sorted(unsupported))}"
    return True, "Every order product can be manufactured by at least one plant"


def check_transport_coverage(
    orders_df: pd.DataFrame, plants_df: pd.DataFrame, transport_df: pd.DataFrame
) -> Tuple[bool, str]:
    """Check that every (plant, destination) pair needed by orders has a transport record."""
    if "product" not in orders_df.columns or "destination" not in orders_df.columns:
        return False, "Cannot verify transport coverage: orders.csv missing required columns"
    if "plant" not in plants_df.columns or "product" not in plants_df.columns:
        return False, "Cannot verify transport coverage: plants.csv missing required columns"
    if "plant" not in transport_df.columns or "destination" not in transport_df.columns:
        return False, "Cannot verify transport coverage: transport_duties.csv missing required columns"

    # Standardize plants and destinations
    plant_products = {}
    for _, row in plants_df.iterrows():
        p = str(row["plant"]).strip().upper()
        prod = str(row["product"]).strip()
        plant_products.setdefault(prod, set()).add(p)

    available_lanes = set()
    for _, row in transport_df.iterrows():
        p = str(row["plant"]).strip().upper()
        d = str(row["destination"]).strip()
        available_lanes.add((p, d))

    # For every order, which (plant, dest) lanes could possibly fulfill it?
    # At least all potential manufacturing plants for that order's product should be able to deliver,
    # or at least the system requires valid lane routes for the product's candidate plants.
    missing_lanes = set()
    for _, row in orders_df.iterrows():
        prod = str(row["product"]).strip()
        dest = str(row["destination"]).strip()
        candidate_plants = plant_products.get(prod, set())
        for plant in candidate_plants:
            if (plant, dest) not in available_lanes:
                missing_lanes.add((plant, dest))

    if missing_lanes:
        sample_missing = sorted(list(missing_lanes))[:3]
        sample_str = ", ".join(f"({p}, {d})" for p, d in sample_missing)
        return False, f"Missing transport record for {sample_str}"
    return True, "All required (plant, destination) transport lanes are covered"


def check_due_dates_after_start(orders_df: pd.DataFrame, planning_start: date) -> Tuple[bool, str]:
    """Check that all order due dates are on or after the planning start date."""
    if "due" not in orders_df.columns:
        return False, "Cannot check due dates: 'due' column missing in orders.csv"

    start_str = planning_start.isoformat()
    past_due_orders = []
    for idx, row in orders_df.iterrows():
        due_val = str(row["due"]).strip()
        try:
            due_date = datetime.strptime(due_val, "%Y-%m-%d").date()
            if due_date < planning_start:
                order_id = row.get("order_id", f"row {idx + 2}")
                past_due_orders.append((order_id, due_val))
        except ValueError:
            continue

    if past_due_orders:
        sample_past = [f"{oid} ({due})" for oid, due in past_due_orders[:3]]
        return False, f"Orders due before planning start ({start_str}): {', '.join(sample_past)}"
    return True, f"All order due dates are on or after planning start ({start_str})"


def validate_orders(df: pd.DataFrame) -> List[Tuple[bool, str]]:
    """Run all validation rules for orders.csv."""
    results: List[Tuple[bool, str]] = []
    # 1. Required columns
    col_ok, col_msg = check_required_columns(df, "orders")
    results.append((col_ok, col_msg))
    if not col_ok:
        return results

    # 2. No missing values in required columns
    results.append(check_no_missing(df, REQUIRED_COLUMNS["orders"], "orders required columns"))

    # 3. Positive integer quantity
    results.append(check_positive_integers(df, "qty", "qty"))

    # 4. Valid due dates format
    results.append(check_valid_dates(df, "due", "due date"))

    # 5. Unique order_id
    results.append(check_duplicates(df, ["order_id"], "order_id"))

    return results


def validate_plants(df: pd.DataFrame) -> List[Tuple[bool, str]]:
    """Run all validation rules for plants.csv."""
    results: List[Tuple[bool, str]] = []
    # 1. Required columns
    col_ok, col_msg = check_required_columns(df, "plants")
    results.append((col_ok, col_msg))
    if not col_ok:
        return results

    # 2. No missing values
    results.append(check_no_missing(df, REQUIRED_COLUMNS["plants"], "plants required columns"))

    # 3. weekly_capacity positive integer
    results.append(check_positive_integers(df, "weekly_capacity", "weekly_capacity"))

    # 4. Non-negative unit_cost
    results.append(check_non_negative(df, "unit_cost", "unit_cost"))

    # 5. Non-negative energy_kwh_per_unit
    results.append(check_non_negative(df, "energy_kwh_per_unit", "energy_kwh_per_unit"))

    # 6. Non-negative carbon_kg_per_kwh
    results.append(check_non_negative(df, "carbon_kg_per_kwh", "carbon_kg_per_kwh"))

    # 7. Unique (plant, product) key
    results.append(check_duplicates(df, ["plant", "product"], "(plant, product)"))

    return results


def validate_transport_duties(df: pd.DataFrame) -> List[Tuple[bool, str]]:
    """Run all validation rules for transport_duties.csv."""
    results: List[Tuple[bool, str]] = []
    # 1. Required columns
    col_ok, col_msg = check_required_columns(df, "transport_duties")
    results.append((col_ok, col_msg))
    if not col_ok:
        return results

    # 2. No missing values
    results.append(check_no_missing(df, REQUIRED_COLUMNS["transport_duties"], "transport_duties required columns"))

    # 3. Non-negative transport_cost_per_unit
    results.append(check_non_negative(df, "transport_cost_per_unit", "transport_cost_per_unit"))

    # 4. transit_days non-negative integer
    # Check non-negative integers (>= 0)
    transit_ok = True
    transit_msg = "All transit_days values are non-negative integers"
    if "transit_days" in df.columns:
        invalid_transit = []
        for idx, val in df["transit_days"].dropna().items():
            try:
                num = float(val)
                if not num.is_integer() or num < 0:
                    invalid_transit.append((idx + 2, val))
            except (ValueError, TypeError):
                invalid_transit.append((idx + 2, val))
        if invalid_transit:
            samples = [f"row {r}: {v}" for r, v in invalid_transit[:3]]
            transit_ok = False
            transit_msg = f"Invalid transit_days (must be integer >= 0): {', '.join(samples)}"
    results.append((transit_ok, transit_msg))

    # 5. duty_pct between 0 and 100
    results.append(check_range(df, "duty_pct", 0.0, 100.0, "duty_pct"))

    # 6. Non-negative carbon_kg_per_unit
    results.append(check_non_negative(df, "carbon_kg_per_unit", "carbon_kg_per_unit"))

    # 7. Unique (plant, destination) key
    results.append(check_duplicates(df, ["plant", "destination"], "(plant, destination)"))

    return results


def validate_capacity_events(df: pd.DataFrame) -> List[Tuple[bool, str]]:
    """Run all validation rules for capacity_events.csv."""
    results: List[Tuple[bool, str]] = []
    # 1. Required columns
    col_ok, col_msg = check_required_columns(df, "capacity_events")
    results.append((col_ok, col_msg))
    if not col_ok:
        return results

    # 2. No missing values
    results.append(check_no_missing(df, REQUIRED_COLUMNS["capacity_events"], "capacity_events required columns"))

    # 3. Valid from_date and to_date
    results.append(check_valid_dates(df, "from_date", "from_date"))
    results.append(check_valid_dates(df, "to_date", "to_date"))

    # 4. Date order to_date >= from_date
    results.append(check_date_order(df, "from_date", "to_date", "capacity_events"))

    # 5. capacity_loss_pct between 0 and 100
    results.append(check_range(df, "capacity_loss_pct", 0.0, 100.0, "capacity_loss_pct"))

    # 6. Unique event_id
    results.append(check_duplicates(df, ["event_id"], "event_id"))

    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Annex B CSV files before database ingestion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        help="Directory containing the CSV files (orders.csv, plants.csv, transport_duties.csv, etc.)",
    )
    parser.add_argument(
        "--planning-start",
        type=str,
        default=None,
        help="Planning start date for due-date checks in YYYY-MM-DD format (default: today)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    data_dir: Path = args.dir

    if not data_dir.is_dir():
        print(f"Error: Provided path is not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine planning start date
    if args.planning_start:
        try:
            planning_start = datetime.strptime(args.planning_start, "%Y-%m-%d").date()
        except ValueError:
            print(f"Error: --planning-start '{args.planning_start}' is not valid YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)
    else:
        planning_start = date.today()

    print("=== Dataset Validation Report ===")

    all_checks: List[Tuple[bool, str]] = []

    # Dictionary to keep loaded dataframes for cross-table validation
    dfs: Dict[str, Optional[pd.DataFrame]] = {
        "orders": None,
        "plants": None,
        "transport_duties": None,
        "capacity_events": None,
    }

    # 1. Orders
    orders_path = data_dir / "orders.csv"
    print(f"\nFile: orders.csv")
    if not orders_path.exists():
        print("  [FAIL] File missing: orders.csv")
        all_checks.append((False, "orders.csv missing"))
    else:
        try:
            df = pd.read_csv(orders_path, dtype=str)
            df = apply_column_aliases(df, "orders")
            dfs["orders"] = df
            results = validate_orders(df)
            for passed, msg in results:
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {msg}")
                all_checks.append((passed, msg))
        except Exception as e:
            print(f"  [FAIL] Could not read orders.csv: {e}")
            all_checks.append((False, f"Read error on orders.csv: {e}"))

    # 2. Plants
    plants_path = data_dir / "plants.csv"
    print(f"\nFile: plants.csv")
    if not plants_path.exists():
        print("  [FAIL] File missing: plants.csv")
        all_checks.append((False, "plants.csv missing"))
    else:
        try:
            df = pd.read_csv(plants_path, dtype=str)
            df = apply_column_aliases(df, "plants")
            dfs["plants"] = df
            results = validate_plants(df)
            for passed, msg in results:
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {msg}")
                all_checks.append((passed, msg))
        except Exception as e:
            print(f"  [FAIL] Could not read plants.csv: {e}")
            all_checks.append((False, f"Read error on plants.csv: {e}"))

    # 3. Transport Duties
    transport_path = data_dir / "transport_duties.csv"
    print(f"\nFile: transport_duties.csv")
    if not transport_path.exists():
        print("  [FAIL] File missing: transport_duties.csv")
        all_checks.append((False, "transport_duties.csv missing"))
    else:
        try:
            df = pd.read_csv(transport_path, dtype=str)
            df = apply_column_aliases(df, "transport_duties")
            dfs["transport_duties"] = df
            results = validate_transport_duties(df)
            for passed, msg in results:
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {msg}")
                all_checks.append((passed, msg))
        except Exception as e:
            print(f"  [FAIL] Could not read transport_duties.csv: {e}")
            all_checks.append((False, f"Read error on transport_duties.csv: {e}"))

    # 4. Capacity Events (Optional)
    events_path = data_dir / "capacity_events.csv"
    print(f"\nFile: capacity_events.csv")
    if not events_path.exists():
        print("  [INFO] File not present (optional)")
    else:
        try:
            df = pd.read_csv(events_path, dtype=str)
            df = apply_column_aliases(df, "capacity_events")
            dfs["capacity_events"] = df
            results = validate_capacity_events(df)
            for passed, msg in results:
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {msg}")
                all_checks.append((passed, msg))
        except Exception as e:
            print(f"  [FAIL] Could not read capacity_events.csv: {e}")
            all_checks.append((False, f"Read error on capacity_events.csv: {e}"))

    # Cross-Table Checks
    print("\nCross-Table Checks:")
    cross_checks: List[Tuple[bool, str]] = []

    # Check 1: Product eligibility
    if dfs["orders"] is not None and dfs["plants"] is not None:
        cross_checks.append(check_product_eligibility(dfs["orders"], dfs["plants"]))
    else:
        cross_checks.append((False, "Skipped product eligibility check: missing orders.csv or plants.csv"))

    # Check 2: Transport coverage
    if dfs["orders"] is not None and dfs["plants"] is not None and dfs["transport_duties"] is not None:
        cross_checks.append(check_transport_coverage(dfs["orders"], dfs["plants"], dfs["transport_duties"]))
    else:
        cross_checks.append((False, "Skipped transport coverage check: missing required tables"))

    # Check 3: Due dates after planning start
    if dfs["orders"] is not None:
        cross_checks.append(check_due_dates_after_start(dfs["orders"], planning_start))
    else:
        cross_checks.append((False, "Skipped due date start check: missing orders.csv"))

    for passed, msg in cross_checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {msg}")
        all_checks.append((passed, msg))

    # Summary
    total = len(all_checks)
    passed_count = sum(1 for passed, _ in all_checks if passed)
    failed_count = total - passed_count
    status_str = "PASS" if failed_count == 0 else "FAIL"

    print("\n---")
    print(f"Total checks: {total}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Status: {status_str}")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
