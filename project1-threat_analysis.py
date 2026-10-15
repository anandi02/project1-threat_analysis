"""
Threat & Security Data Analysis Project
---------------------------------------
A starter Python project for:
1. Loading and cleaning cybersecurity data
2. Extracting Indicators of Compromise (IoCs)
3. Basic threat statistics and timeline analysis
4. Machine-learning classification
5. Feature-importance / XAI analysis
6. Saving analysis outputs

Usage:
    python threat_analysis.py --input your_data.csv

If no input file is supplied, the script creates a small demo dataset so
the project can be tested immediately.

Expected CSV columns (flexible):
    timestamp, source_ip, destination_ip, protocol, bytes,
    failed_logins, malware, severity, label

The script automatically uses the columns that are available.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Optional ML imports. Install with:
# pip install pandas numpy scikit-learn matplotlib seaborn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def create_demo_data() -> pd.DataFrame:
    """Create a small cybersecurity dataset for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", periods=20, freq="6h"
            ),
            "source_ip": [
                "192.168.1.10", "10.0.0.5", "185.220.101.1", "192.168.1.20",
                "45.33.32.156", "10.0.0.8", "185.220.101.1", "172.16.0.4",
                "45.33.32.156", "192.168.1.10", "8.8.8.8", "10.0.0.5",
                "185.220.101.1", "172.16.0.4", "45.33.32.156", "10.0.0.8",
                "192.168.1.20", "185.220.101.1", "8.8.8.8", "10.0.0.5",
            ],
            "destination_ip": [
                "10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4",
                "10.0.0.5", "10.0.0.6", "10.0.0.7", "10.0.0.8",
                "10.0.0.9", "10.0.0.10", "10.0.0.11", "10.0.0.12",
                "10.0.0.13", "10.0.0.14", "10.0.0.15", "10.0.0.16",
                "10.0.0.17", "10.0.0.18", "10.0.0.19", "10.0.0.20",
            ],
            "protocol": [
                "TCP", "TCP", "TCP", "UDP", "TCP",
                "TCP", "UDP", "TCP", "TCP", "UDP",
                "TCP", "TCP", "TCP", "UDP", "TCP",
                "TCP", "UDP", "TCP", "TCP", "UDP",
            ],
            "bytes": [
                1200, 2300, 95000, 800, 87000,
                1400, 120000, 900, 110000, 700,
                1300, 2100, 99000, 750, 92000,
                1600, 105000, 650, 98000, 1700,
            ],
            "failed_logins": [
                0, 1, 12, 0, 18, 1, 15, 0, 22, 0,
                1, 2, 14, 0, 20, 16, 0, 19, 1, 2,
            ],
            "malware": [
                0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 0, 1, 0, 1, 1, 0, 1, 0, 0,
            ],
            "severity": [
                "low", "low", "high", "low", "high",
                "medium", "high", "low", "high", "low",
                "low", "medium", "high", "low", "high",
                "high", "low", "high", "low", "medium",
            ],
            "label": [
                0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 0, 1, 0, 1, 1, 0, 1, 0, 0,
            ],
        }
    )


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_data(input_path: str | None) -> pd.DataFrame:
    """Load CSV data, or use demo data when no file is supplied."""
    if not input_path:
        print("[INFO] No input supplied. Using demo dataset.")
        return create_demo_data()

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Supported input formats: CSV and Excel.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform safe, generic cleaning."""
    data = df.copy()
    data.columns = [
        re.sub(r"[^a-zA-Z0-9_]+", "_", str(c).strip().lower()).strip("_")
        for c in data.columns
    ]

    # Remove exact duplicate rows.
    data = data.drop_duplicates().reset_index(drop=True)

    # Parse timestamp when present.
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(
            data["timestamp"], errors="coerce"
        )

    # Normalize common text fields.
    for column in ["protocol", "severity", "malware"]:
        if column in data.columns:
            data[column] = data[column].astype(str).str.strip().str.lower()

    return data


# ---------------------------------------------------------------------------
# IoC extraction
# ---------------------------------------------------------------------------

IP_PATTERN = re.compile(
    r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"
)

DOMAIN_PATTERN = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[A-Za-z]{2,63}\b"
)

HASH_PATTERN = re.compile(
    r"\b[a-fA-F0-9]{32}\b|"
    r"\b[a-fA-F0-9]{40}\b|"
    r"\b[a-fA-F0-9]{64}\b"
) 


def extract_iocs(values: Iterable[object]) -> dict[str, list[str]]:
    """Extract common IoC types from arbitrary text values."""
    ips: set[str] = set()
    domains: set[str] = set()
    hashes: set[str] = set()

    for value in values:
        text = str(value)
        ips.update(IP_PATTERN.findall(text))
        domains.update(DOMAIN_PATTERN.findall(text))
        hashes.update(HASH_PATTERN.findall(text))

    return {
        "ip_addresses": sorted(ips),
        "domains": sorted(domains),
        "hashes": sorted(hashes),
    }


def extract_iocs_from_dataframe(df: pd.DataFrame) -> dict[str, list[str]]:
    """Extract IoCs from all textual/object columns."""
    values = df.astype(str).to_numpy().ravel()
    return extract_iocs(values)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate basic dataset statistics."""
    rows = len(df)
    columns = len(df.columns)
    missing = int(df.isna().sum().sum())

    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "missing_values",
                "duplicate_rows_after_cleaning",
            ],
            "value": [
                rows,
                columns,
                missing,
                int(df.duplicated().sum()),
            ],
        }
    )
    return summary


def threat_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return useful threat-related statistics when columns exist."""
    records = []

    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="coerce")
        records.append(
            {
                "metric": "labelled_threat_events",
                "value": int((labels == 1).sum()),
            }
        )

    if "severity" in df.columns:
        records.append(
            {
                "metric": "high_severity_events",
                "value": int(
                    (df["severity"].astype(str).str.lower() == "high").sum()
                ),
            }
        )

    if "failed_logins" in df.columns:
        failed = pd.to_numeric(df["failed_logins"], errors="coerce")
        records.append(
            {
                "metric": "total_failed_logins",
                "value": float(failed.fillna(0).sum()),
            }
        )

    if "source_ip" in df.columns:
        records.append(
            {
                "metric": "unique_source_ips",
                "value": int(df["source_ip"].nunique()),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------------

def prepare_ml_data(df: pd.DataFrame):
    """
    Prepare a generic binary classification problem.

    The target column is 'label'. Timestamp is converted to useful numeric
    components instead of being passed directly to the model.
    """
    if "label" not in df.columns:
        print("[INFO] No 'label' column found. ML step skipped.")
        return None

    data = df.copy()
    y = pd.to_numeric(data["label"], errors="coerce")
    valid = y.notna()

    data = data.loc[valid].copy()
    y = y.loc[valid].astype(int)

    if y.nunique() < 2:
        print("[INFO] Target contains fewer than two classes. ML skipped.")
        return None

    data = data.drop(columns=["label"])

    if "timestamp" in data.columns:
        ts = pd.to_datetime(data["timestamp"], errors="coerce")
        data["hour"] = ts.dt.hour.fillna(-1)
        data["day_of_week"] = ts.dt.dayofweek.fillna(-1)
        data = data.drop(columns=["timestamp"])

    # Convert obvious numeric-looking object columns where possible.
    for col in data.columns:
        if data[col].dtype == "object":
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().mean() > 0.8:
                data[col] = converted

    numeric_cols = data.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_cols = [
        c for c in data.columns if c not in numeric_cols
    ]

    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_cols,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", transformer),
            ("model", model),
        ]
    )

    return data, y, pipeline


def run_ml(df: pd.DataFrame, output_dir: Path) -> None:
    """Train a Random Forest model and save evaluation results."""
    prepared = prepare_ml_data(df)
    if prepared is None:
        return

    X, y, pipeline = prepared

    if len(X) < 10:
        print("[INFO] Too few rows for a useful train/test split. ML skipped.")
        return

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    print("\n=== ML Results ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(report)

    # Save confusion matrix.
    cm = confusion_matrix(y_test, predictions)
    pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(output_dir / "confusion_matrix.csv")

    # Save classification report.
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    # Permutation importance provides model-agnostic feature importance.
    try:
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="accuracy",
        )

        importance = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

        importance.to_csv(
            output_dir / "feature_importance.csv",
            index=False,
        )
    except Exception as exc:
        print(f"[WARNING] Feature importance could not be calculated: {exc}")


# ---------------------------------------------------------------------------
# Timeline analysis
# ---------------------------------------------------------------------------

def timeline_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a simple event-count timeline."""
    if "timestamp" not in df.columns:
        print("[INFO] No timestamp column. Timeline skipped.")
        return

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"])

    if data.empty:
        print("[INFO] No valid timestamps. Timeline skipped.")
        return

    timeline = (
        data.set_index("timestamp")
        .resample("D")
        .size()
        .rename("event_count")
        .reset_index()
    )

    timeline.to_csv(output_dir / "daily_timeline.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cybersecurity threat and data analysis project."
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to CSV or Excel input data.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="threat_analysis_output",
        help="Output directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Threat & Security Data Analysis ===")

    # 1. Load and clean.
    df = load_data(args.input)
    print(f"[INFO] Loaded {len(df)} rows and {len(df.columns)} columns.")

    df = clean_data(df)
    print(f"[INFO] After cleaning: {len(df)} rows.")

    # 2. Save cleaned data.
    df.to_csv(output_dir / "cleaned_data.csv", index=False)

    # 3. Summary.
    generate_summary(df).to_csv(
        output_dir / "dataset_summary.csv",
        index=False,
    )

    stats = threat_statistics(df)
    if not stats.empty:
        stats.to_csv(
            output_dir / "threat_statistics.csv",
            index=False,
        )

    # 4. IoCs.
    iocs = extract_iocs_from_dataframe(df)

    for name, values in iocs.items():
        pd.DataFrame({"ioc": values}).to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )

    print("\n=== IoC Summary ===")
    for name, values in iocs.items():
        print(f"{name}: {len(values)}")

    # 5. Timeline.
    timeline_analysis(df, output_dir)

    # 6. Machine learning + feature importance.
    run_ml(df, output_dir)

    print(f"\n[DONE] Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
