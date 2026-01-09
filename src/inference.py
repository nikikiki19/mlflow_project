from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence

import mlflow
from pyspark.ml import PipelineModel
from pyspark.ml.feature import ImputerModel, StringIndexerModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, concat_ws, trim, when
from pyspark.sql.types import DoubleType, StringType

from src.train import _normalize_dataframe_columns, _normalize_col_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using a registered MLflow Spark model.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to input CSV.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/predictions"),
        help="Output directory for predictions (Spark writes a folder).",
    )
    parser.add_argument("--model-name", type=str, default="Churn_Model", help="Registered model name.")
    parser.add_argument("--model-stage", type=str, default="Production", help="Model stage to load.")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--id-columns",
        nargs="*",
        default=["CustomerID"],
        help="ID columns to keep in output if present.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Churn",
        help="Optional label column to keep if present.",
    )
    parser.add_argument(
        "--keep-input",
        action="store_true",
        help="Keep all input columns in the output alongside predictions.",
    )
    return parser.parse_args()


def start_spark(app_name: str) -> SparkSession:
    spark = SparkSession.builder.appName(app_name).config("spark.ui.showConsoleProgress", "false").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_dataframe(spark: SparkSession, path: Path) -> DataFrame:
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .option("nullValue", "NA")
        .option("mode", "DROPMALFORMED")
        .csv(str(path))
    )


def _collect_expected_columns(model: PipelineModel) -> tuple[List[str], List[str]]:
    categorical_cols: List[str] = []
    numeric_cols: List[str] = []
    for stage in model.stages:
        if isinstance(stage, StringIndexerModel):
            categorical_cols.append(stage.getInputCol())
        elif isinstance(stage, ImputerModel):
            numeric_cols.extend(stage.getInputCols())
    return sorted(set(categorical_cols)), sorted(set(numeric_cols))


def _prepare_inputs(
    df: DataFrame,
    categorical_cols: Sequence[str],
    numeric_cols: Sequence[str],
) -> DataFrame:
    missing = [c for c in list(categorical_cols) + list(numeric_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in inference data: {missing}")

    for col_name in categorical_cols:
        df = df.withColumn(col_name, col(col_name).cast(StringType()))
        df = df.withColumn(col_name, when(trim(col(col_name)) == "", None).otherwise(col(col_name)))
    if categorical_cols:
        df = df.fillna("missing", subset=list(categorical_cols))

    for col_name in numeric_cols:
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

    return df


def _select_output_columns(
    df: DataFrame,
    id_cols: Sequence[str],
    label_col: str,
    keep_input: bool,
) -> List[str]:
    if keep_input:
        base_cols = df.columns
    else:
        base_cols = [c for c in id_cols if c in df.columns]
        if label_col and label_col in df.columns:
            base_cols.append(label_col)
    out_cols = base_cols + [c for c in ["prediction", "probability"] if c in df.columns]
    if not out_cols:
        out_cols = df.columns
    return out_cols


def _coerce_probability_column(df: DataFrame) -> DataFrame:
    if "probability" not in df.columns:
        return df
    return df.withColumn("probability", concat_ws(",", vector_to_array(col("probability"))))


def main() -> None:
    args = parse_args()
    if not args.data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.data_path}")

    mlflow.set_tracking_uri(args.tracking_uri)
    model_uri = f"models:/{args.model_name}/{args.model_stage}"

    spark = start_spark(app_name="cell2cell-churn-inference")
    try:
        model = mlflow.spark.load_model(model_uri)
        df = load_dataframe(spark, args.data_path)
        df = _normalize_dataframe_columns(df)

        id_cols = [_normalize_col_name(c) for c in (args.id_columns or []) if _normalize_col_name(c)]
        label_col = _normalize_col_name(args.label_column)

        categorical_cols, numeric_cols = _collect_expected_columns(model)
        df_prepped = _prepare_inputs(df, categorical_cols, numeric_cols)

        pred_df = model.transform(df_prepped)
        pred_df = _coerce_probability_column(pred_df)
        out_cols = _select_output_columns(pred_df, id_cols, label_col, args.keep_input)

        (
            pred_df.select(*out_cols)
            .write.mode("overwrite")
            .option("header", "true")
            .csv(str(args.output_path))
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
