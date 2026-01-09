from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send inference requests to MLflow model server.")
    parser.add_argument("--data-path", type=Path, required=True, help="CSV file with input rows.")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:5002/invocations",
        help="MLflow model server /invocations endpoint.",
    )
    parser.add_argument("--n-rows", type=int, default=1, help="How many rows to send.")
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["Churn"],
        help="Columns to drop before sending (e.g., label).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.data_path}")

    df = pd.read_csv(args.data_path).head(args.n_rows)
    if args.drop_columns:
        df = df.drop(columns=args.drop_columns, errors="ignore")

    payload = {"dataframe_split": df.to_dict(orient="split")}
    response = requests.post(args.server_url, json=payload, timeout=60)
    print(response.status_code)
    print(response.text)


if __name__ == "__main__":
    main()
