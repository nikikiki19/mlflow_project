from __future__ import annotations

import argparse
import os
import tempfile
from itertools import chain
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, create_map, lit, trim, when
from pyspark.sql.types import DoubleType, StringType

from src.data_validation import validate_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Cell2Cell churn model with PySpark + MLflow.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to Cell2Cell CSV.")
    parser.add_argument("--label-column", type=str, default="Churn", help="Name of the label column.")
    parser.add_argument(
        "--id-columns",
        nargs="*",
        default=["CustomerID"],
        help="ID columns to exclude from features; kept in predictions for traceability.",
    )
    parser.add_argument("--experiment-name", type=str, default="cell2cell-churn", help="MLflow experiment name.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional MLflow run name.")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
        help="MLflow tracking URI.",
    )
    parser.add_argument("--register-model-name", type=str, default=None, help="Register model under this name if set.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits and model.")
    parser.add_argument("--shuffle-partitions", type=int, default=200, help="spark.sql.shuffle.partitions.")
    parser.add_argument("--max-iter", type=int, default=60, help="Number of boosting iterations.")
    parser.add_argument("--max-depth", type=int, default=5, help="Tree depth.")
    parser.add_argument("--max-bins", type=int, default=64, help="Max bins for continuous features.")
    parser.add_argument("--step-size", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--subsampling-rate", type=float, default=0.8, help="Subsampling rate for each iteration.")
    parser.add_argument("--min-info-gain", type=float, default=0.0, help="Minimum info gain for a split.")
    parser.add_argument("--use-class-weights", action="store_true", help="Apply class weights to handle imbalance.")
    parser.add_argument(
        "--prediction-sample-size",
        type=int,
        default=200,
        help="How many prediction rows to log as a sample artifact.",
    )

    args = parser.parse_args()
    if not args.data_path.exists():
        parser.error(f"Data path not found: {args.data_path}")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")
    if args.prediction_sample_size <= 0:
        parser.error("--prediction-sample-size must be > 0.")
    return args


def start_spark(app_name: str, shuffle_partitions: int) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
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


def _normalize_col_name(x) -> str:
    return str(x or "").replace("\ufeff", "").strip()


def _normalize_dataframe_columns(df: DataFrame) -> DataFrame:
    original = df.columns
    normalized = [_normalize_col_name(c) for c in original]

    pairs = list(zip(original, normalized))

    to_drop = [old for old, new in pairs if new == ""]
    if to_drop:
        df = df.drop(*to_drop)
        pairs = [(old, new) for (old, new) in pairs if new != ""]

    seen = set()
    dedup_pairs = []
    drop_dupes = []
    for old, new in pairs:
        if new in seen:
            drop_dupes.append(old)
        else:
            seen.add(new)
            dedup_pairs.append((old, new))

    if drop_dupes:
        df = df.drop(*drop_dupes)
        dedup_pairs = [(old, new) for (old, new) in dedup_pairs if old in df.columns]

    for old, new in dedup_pairs:
        if old != new and old in df.columns:
            df = df.withColumnRenamed(old, new)

    return df


def _sanitize_feature_lists(
    df: DataFrame,
    categorical_cols: Sequence,
    numeric_cols: Sequence,
    label_col: str,
    id_cols: Sequence[str],
) -> Tuple[List[str], List[str]]:
    df_cols_set = set(df.columns)

    def clean_list(seq) -> List[str]:
        out = []
        for x in seq or []:
            s = _normalize_col_name(x)
            if not s:
                continue
            if s not in df_cols_set:
                continue
            out.append(s)
        seen = set()
        uniq = []
        for c in out:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    cat = clean_list(categorical_cols)
    num = clean_list(numeric_cols)

    exclude = set([_normalize_col_name(label_col), "label", "features"])
    exclude |= set(_normalize_col_name(c) for c in (id_cols or []))

    cat = [c for c in cat if c not in exclude]
    num = [c for c in num if c not in exclude]

    return cat, num


def preprocess_dataframe(
    df: DataFrame, label_col: str, id_cols: Sequence[str], categorical_cols: List[str], numeric_cols: List[str]
) -> DataFrame:
    for col_name in categorical_cols:
        df = df.withColumn(col_name, col(col_name).cast(StringType()))
        df = df.withColumn(
            col_name,
            when(trim(col(col_name)) == "", None).otherwise(col(col_name)),
        )
    if categorical_cols:
        df = df.fillna("missing", subset=categorical_cols)

    for col_name in numeric_cols:
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

    df = df.dropna(subset=[label_col])

    if id_cols:
        present_ids = [c for c in id_cols if c in df.columns]
        if present_ids:
            df = df.dropDuplicates(subset=present_ids + [label_col])

    return df


def index_label(df: DataFrame, label_col: str) -> Tuple[DataFrame, List[str]]:
    label_indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="error")
    model = label_indexer.fit(df)
    indexed = model.transform(df)
    return indexed, list(model.labels)


def add_class_weights(df: DataFrame, label_col: str) -> Tuple[DataFrame, Dict[float, float]]:
    counts = df.groupBy(label_col).count().collect()
    total = sum(row["count"] for row in counts)
    num_classes = len(counts)
    weights = {float(row[label_col]): total / (num_classes * row["count"]) for row in counts}
    mapping_expr = create_map([lit(x) for x in chain.from_iterable(weights.items())])
    weighted_df = df.withColumn("class_weight", mapping_expr[col(label_col)])
    return weighted_df, weights


def build_pipeline(categorical_cols: List[str], numeric_cols: List[str], args: argparse.Namespace) -> Pipeline:
    stages = []

    encoded_cols: List[str] = []
    if categorical_cols:
        index_output_cols = [f"{c}_idx" for c in categorical_cols]
        stages.extend(
            [
                StringIndexer(inputCol=c, outputCol=idx_col, handleInvalid="keep")
                for c, idx_col in zip(categorical_cols, index_output_cols)
            ]
        )
        encoded_cols = [f"{c}_ohe" for c in categorical_cols]
        stages.append(
            OneHotEncoder(
                inputCols=index_output_cols,
                outputCols=encoded_cols,
                handleInvalid="keep",
            )
        )

    imputed_numeric_cols: List[str] = []
    if numeric_cols:
        imputed_numeric_cols = [f"{c}_imputed" for c in numeric_cols]
        stages.append(
            Imputer(
                strategy="median",
                inputCols=numeric_cols,
                outputCols=imputed_numeric_cols,
            )
        )

    feature_inputs = encoded_cols + (imputed_numeric_cols or numeric_cols)
    feature_inputs = [c for c in feature_inputs if c and _normalize_col_name(c)]
    if not feature_inputs:
        raise ValueError("No feature columns after sanitization. Check CSV headers and validation outputs.")

    stages.append(VectorAssembler(inputCols=feature_inputs, outputCol="features"))

    gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        maxIter=args.max_iter,
        maxDepth=args.max_depth,
        maxBins=args.max_bins,
        stepSize=args.step_size,
        subsamplingRate=args.subsampling_rate,
        minInfoGain=args.min_info_gain,
        seed=args.seed,
    )
    if args.use_class_weights:
        gbt = gbt.setParams(weightCol="class_weight")

    stages.append(gbt)
    return Pipeline(stages=stages)


def evaluate(pred_df: DataFrame) -> Dict[str, float]:
    bc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    mc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    return {
        "roc_auc": bc.evaluate(pred_df),
        "accuracy": mc.setMetricName("accuracy").evaluate(pred_df),
        "f1": mc.setMetricName("f1").evaluate(pred_df),
    }


def log_prediction_sample(pred_df: DataFrame, label_col: str, id_cols: Sequence[str], sample_size: int) -> None:
    keep_cols = [c for c in id_cols if c in pred_df.columns] + [label_col, "prediction", "probability"]
    sample = pred_df.select(*keep_cols).limit(sample_size)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "prediction_sample.csv"
        sample.toPandas().to_csv(out_path, index=False)
        mlflow.log_artifact(str(out_path), artifact_path="predictions")


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    spark = start_spark(app_name="cell2cell-churn-train", shuffle_partitions=args.shuffle_partitions)

    try:
        df_raw = load_dataframe(spark, args.data_path)
        df_raw = _normalize_dataframe_columns(df_raw)

        label_col = _normalize_col_name(args.label_column)
        id_cols = [_normalize_col_name(c) for c in (args.id_columns or []) if _normalize_col_name(c)]

        if not label_col or label_col not in df_raw.columns:
            raise ValueError(
                f"Label column '{args.label_column}' not found after normalization. Available: {df_raw.columns[:30]}"
            )

        cat_cols_raw, num_cols_raw = validate_dataframe(df_raw, label_col, id_cols)
        categorical_cols, numeric_cols = _sanitize_feature_lists(df_raw, cat_cols_raw, num_cols_raw, label_col, id_cols)

        df_prepped = preprocess_dataframe(df_raw, label_col, id_cols, categorical_cols, numeric_cols)
        df_indexed, label_values = index_label(df_prepped, label_col)

        if args.use_class_weights:
            df_for_train, class_weights = add_class_weights(df_indexed, "label")
        else:
            df_for_train, class_weights = df_indexed, None

        train_df, test_df = df_for_train.randomSplit([1 - args.test_size, args.test_size], seed=args.seed)
        pipeline = build_pipeline(categorical_cols, numeric_cols, args)

        with mlflow.start_run(run_name=args.run_name):
            mlflow.set_tags({"dataset": "Cell2Cell", "model": "GBTClassifier", "framework": "pyspark"})

            mlflow.log_params(
                {
                    "data_path": str(args.data_path),
                    "label_column": label_col,
                    "id_columns": ",".join(id_cols) if id_cols else "",
                    "test_size": args.test_size,
                    "max_iter": args.max_iter,
                    "max_depth": args.max_depth,
                    "max_bins": args.max_bins,
                    "step_size": args.step_size,
                    "subsampling_rate": args.subsampling_rate,
                    "min_info_gain": args.min_info_gain,
                    "use_class_weights": args.use_class_weights,
                    "shuffle_partitions": args.shuffle_partitions,
                    "categorical_cols_count": len(categorical_cols),
                    "numeric_cols_count": len(numeric_cols),
                }
            )

            mlflow.log_dict({"label_index": label_values}, "label_index.json")
            if class_weights:
                mlflow.log_dict({"class_weights": class_weights}, "class_weights.json")

            model = pipeline.fit(train_df)

            predictions = model.transform(test_df).cache()
            metrics = evaluate(predictions)
            mlflow.log_metrics(metrics)

            log_prediction_sample(predictions, label_col, id_cols, args.prediction_sample_size)

            mlflow.spark.log_model(
                model,
                artifact_path="model",
                registered_model_name=args.register_model_name,
            )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
