from __future__ import annotations

from typing import List, Sequence, Tuple

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import BooleanType, NumericType, StringType


def detect_column_types(
    df: DataFrame, label_col: str, id_cols: Sequence[str] | None = None
) -> Tuple[List[str], List[str]]:
    id_cols = set(id_cols or [])
    categorical: List[str] = []
    numeric: List[str] = []

    for field in df.schema.fields:
        name = field.name
        if name == label_col or name in id_cols:
            continue
        dtype = field.dataType
        if isinstance(dtype, StringType):
            categorical.append(name)
        elif isinstance(dtype, (NumericType, BooleanType)):
            numeric.append(name)
        else:
            categorical.append(name)

    return categorical, numeric


def validate_dataframe(
    df: DataFrame, label_col: str, id_cols: Sequence[str] | None = None
) -> Tuple[List[str], List[str]]:
    missing_columns = [label_col] if label_col not in df.columns else []
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    id_cols = set(id_cols or [])
    row_count = df.count()
    if row_count == 0:
        raise ValueError("Dataset is empty.")

    null_labels = df.filter(col(label_col).isNull()).count()
    if null_labels > 0:
        raise ValueError(f"Found {null_labels} rows with null labels in '{label_col}'.")

    distinct_labels = df.select(label_col).distinct().count()
    if distinct_labels < 2:
        raise ValueError(f"Need at least two label classes, found {distinct_labels}.")

    categorical, numeric = detect_column_types(df, label_col, id_cols=id_cols)
    if not categorical and not numeric:
        raise ValueError("No feature columns detected after excluding label/id columns.")

    return categorical, numeric
