import time
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error


def build_dataset_path(frequency: str, weather_variant: str):
    """
    Build the CSV path for the training dataset.

    Path pattern:
    - with weather variant: `{frequency}_{weather_variant}/training_dataset_{frequency}_{weather_variant}.csv`
    - without weather variant: `{frequency}/training_dataset_{frequency}.csv`

    Args:
        frequency: Dataset frequency, such as `hourly` or `monthly`.
        weather_variant: Weather configuration suffix, such as
            `with_weather`, `without_weather`, or `weather_null`.
            Use an empty string when no weather-specific suffix is used.

    Returns:
        The relative path to the training dataset CSV file.
    """
    if weather_variant:
        return (
            f"{frequency}_{weather_variant}/"
            f"training_dataset_{frequency}_{weather_variant}.csv"
        )
    return f"{frequency}/training_dataset_{frequency}.csv"


def build_artifact_directory(frequency: str, weather_variant: str):
    """
    Build the directory path used for saved model artifacts.

    Directory pattern:
    - with weather variant: `{frequency}_{weather_variant}`
    - without weather variant: `{frequency}`

    Args:
        frequency: Dataset frequency, such as `hourly` or `monthly`.
        weather_variant: Weather configuration suffix. Use an empty string
            when no weather-specific suffix is used.

    Returns:
        A `Path` object pointing to the artifact directory.
    """
    if weather_variant:
        return Path(f"{frequency}_{weather_variant}")
    return Path(frequency)


def load_training_dataset(csv_path: str):
    """
    Load a training dataset from CSV.

    Args:
        csv_path: Path to the input CSV file.

    Returns:
        A pandas DataFrame containing the raw training data.
    """
    return pd.read_csv(csv_path, low_memory=False)


def preprocess_training_dataset(df: pd.DataFrame):
    """
    Apply basic preprocessing to the training dataset.

    This transformation:
    - converts `timestamp_utc` to timezone-aware datetime
    - sorts rows by `group_id` and `timestamp_utc`

    Args:
        df: Raw training DataFrame.

    Returns:
        A preprocessed pandas DataFrame.
    """
    processed_df = df.copy()
    processed_df["timestamp_utc"] = pd.to_datetime(
        processed_df["timestamp_utc"],
        utc=True,
    )
    processed_df = processed_df.sort_values(
        ["group_id", "timestamp_utc"]
    ).reset_index(drop=True)

    return processed_df


def get_categorical_feature_columns(include_weather_feature: bool):
    """
    Define the categorical feature columns used for LightGBM training.

    Args:
        include_weather_feature: Whether `weather_key` should be included
            in the categorical feature list.

    Returns:
        A list of categorical feature column names.
    """
    base_columns = [
        "group_id",
        "desc",
        "macro_region",
        "region",
        "municipality",
        "segment",
        "product_type",
        "consumption_bucket",
    ]

    if include_weather_feature:
        return ["weather_key"] + base_columns

    return base_columns


def cast_columns_to_category(df: pd.DataFrame, categorical_columns: list[str]):
    """
    Cast available categorical columns to pandas `category` dtype.

    Args:
        df: Input pandas DataFrame.
        categorical_columns: Candidate categorical column names.

    Returns:
        A DataFrame where available categorical columns are cast to `category`.
    """
    processed_df = df.copy()

    for column_name in categorical_columns:
        if column_name in processed_df.columns:
            processed_df[column_name] = processed_df[column_name].astype("category")

    return processed_df


def get_feature_columns(df: pd.DataFrame, target_column: str = "target_consumption"):
    """
    Derive the model feature column list.

    The target column and `timestamp_utc` are excluded from the feature set.

    Args:
        df: Input pandas DataFrame.
        target_column: Name of the target variable column.

    Returns:
        A list of feature column names.
    """
    excluded_columns = [target_column, "timestamp_utc"]
    return [column for column in df.columns if column not in excluded_columns]


def get_validation_cutoff_timestamp(frequency: str):
    """
    Get the time-based validation cutoff for the selected dataset frequency.

    Current cutoffs:
    - hourly: 2024-09-15 00:00:00 UTC
    - monthly: 2023-10-01 00:00:00 UTC

    Args:
        frequency: Dataset frequency, such as `hourly` or `monthly`.

    Returns:
        A timezone-aware pandas Timestamp.

    Raises:
        ValueError: If the frequency is unsupported.
    """
    if frequency == "hourly":
        return pd.Timestamp("2024-09-15 00:00:00", tz="UTC")
    if frequency == "monthly":
        return pd.Timestamp("2023-10-01 00:00:00", tz="UTC")

    raise ValueError(f"Unsupported frequency: {frequency}")


def split_train_and_validation_sets(df: pd.DataFrame, feature_columns: list[str], cutoff_timestamp: pd.Timestamp, target_column: str = "target_consumption"):
    """
    Split the dataset into time-based training and validation sets.

    Rows earlier than `cutoff_timestamp` are used for training.
    Rows at or after `cutoff_timestamp` are used for validation.

    Args:
        df: Input pandas DataFrame.
        feature_columns: Feature column names.
        cutoff_timestamp: Timestamp used for the time-based split.
        target_column: Name of the target variable column.

    Returns:
        A tuple containing:
        - X_train
        - y_train
        - X_valid
        - y_valid
    """
    train_df = df[df["timestamp_utc"] < cutoff_timestamp].copy()
    valid_df = df[df["timestamp_utc"] >= cutoff_timestamp].copy()

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_valid = valid_df[feature_columns]
    y_valid = valid_df[target_column]

    return X_train, y_train, X_valid, y_valid


def create_lightgbm_regressor():
    """
    Create a LightGBM regressor with the configured hyperparameters.

    Returns:
        A configured `lightgbm.LGBMRegressor` instance.
    """
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def train_validation_model(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series, categorical_columns: list[str]):
    """
    Train a validation model with early stopping and measure training time.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_valid: Validation feature matrix.
        y_valid: Validation target vector.
        categorical_columns: Categorical feature column names.

    Returns:
        A tuple containing:
        - the trained validation model
        - training duration in seconds
    """
    model = create_lightgbm_regressor()

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        categorical_feature=categorical_columns,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    training_duration_seconds = time.time() - start_time
    return model, training_duration_seconds


def evaluate_validation_mae(model: lgb.LGBMRegressor, X_valid: pd.DataFrame, y_valid: pd.Series):
    """
    Evaluate a trained validation model using mean absolute error.

    Args:
        model: Trained validation model.
        X_valid: Validation feature matrix.
        y_valid: Validation target vector.

    Returns:
        Validation MAE.
    """
    validation_predictions = model.predict(X_valid)
    return mean_absolute_error(y_valid, validation_predictions)


def train_final_model_on_full_data(df: pd.DataFrame, feature_columns: list[str], categorical_columns: list[str], target_column: str = "target_consumption"):
    """
    Train the final LightGBM model on the full dataset.

    Args:
        df: Full preprocessed training DataFrame.
        feature_columns: Feature column names used for training.
        categorical_columns: Categorical feature column names.
        target_column: Name of the target variable column.

    Returns:
        A tuple containing:
        - the trained final model
        - training duration in seconds
    """
    final_model = create_lightgbm_regressor()

    X_all = df[feature_columns]
    y_all = df[target_column]

    start_time = time.time()

    final_model.fit(
        X_all,
        y_all,
        categorical_feature=categorical_columns,
    )

    training_duration_seconds = time.time() - start_time
    return final_model, training_duration_seconds


def save_model_artifacts(model: lgb.LGBMRegressor, feature_columns: list[str], categorical_columns: list[str], artifact_directory: Path, frequency: str, weather_variant: str):
    """
    Save the final model and training metadata to disk.

    Saved artifacts:
    - trained LightGBM model
    - feature column list
    - categorical column list

    File naming pattern:
    - with weather variant:
      - `lightgbm_fortum_model_{frequency}_{weather_variant}.pkl`
      - `feature_cols_{frequency}_{weather_variant}.pkl`
      - `categorical_cols_{frequency}_{weather_variant}.pkl`
    - without weather variant:
      - `lightgbm_fortum_model_{frequency}.pkl`
      - `feature_cols_{frequency}.pkl`
      - `categorical_cols_{frequency}.pkl`

    Args:
        model: Final trained LightGBM model.
        feature_columns: Feature column names used during training.
        categorical_columns: Categorical column names used during training.
        artifact_directory: Directory where artifacts will be saved.
        frequency: Dataset frequency, such as `hourly` or `monthly`.
        weather_variant: Weather configuration suffix used in file names.

    Returns:
        None.

    Side Effects:
        Creates the target directory if needed and writes pickle files to disk.
    """
    artifact_directory.mkdir(parents=True, exist_ok=True)

    if weather_variant:
        model_filename = f"lightgbm_fortum_model_{frequency}_{weather_variant}.pkl"
        feature_filename = f"feature_cols_{frequency}_{weather_variant}.pkl"
        categorical_filename = f"categorical_cols_{frequency}_{weather_variant}.pkl"
    else:
        model_filename = f"lightgbm_fortum_model_{frequency}.pkl"
        feature_filename = f"feature_cols_{frequency}.pkl"
        categorical_filename = f"categorical_cols_{frequency}.pkl"

    joblib.dump(model, artifact_directory / model_filename)
    joblib.dump(feature_columns, artifact_directory / feature_filename)
    joblib.dump(categorical_columns, artifact_directory / categorical_filename)


def train_lightgbm_model(frequency: str, weather_variant: str, include_weather_feature: bool):
    """
    Train and save a LightGBM forecasting model.

    The workflow:
    - loads the training dataset from CSV
    - preprocesses timestamps and sort order
    - casts selected columns to categorical dtype
    - creates a time-based train/validation split
    - trains a validation model with early stopping
    - evaluates validation MAE
    - trains a final model on the full dataset
    - saves the final model and metadata artifacts

    Args:
        frequency: Dataset frequency, such as `hourly` or `monthly`.
        weather_variant: Weather configuration identifier used in the
            dataset path and artifact naming, such as `weather_null`,
            `with_weather`, or `without_weather`. Use an empty string
            for monthly training.
        include_weather_feature: Whether `weather_key` should be included
            in the categorical feature list.

    Returns:
        None.
    """
    dataset_path = build_dataset_path(frequency, weather_variant)
    artifact_directory = build_artifact_directory(frequency, weather_variant)

    df = load_training_dataset(dataset_path)
    df = preprocess_training_dataset(df)

    categorical_columns = get_categorical_feature_columns(include_weather_feature)
    df = cast_columns_to_category(df, categorical_columns)

    feature_columns = get_feature_columns(df)
    cutoff_timestamp = get_validation_cutoff_timestamp(frequency)

    X_train, y_train, X_valid, y_valid = split_train_and_validation_sets(
        df,
        feature_columns,
        cutoff_timestamp,
    )

    validation_model, validation_training_seconds = train_validation_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_columns,
    )

    validation_mae = evaluate_validation_mae(
        validation_model,
        X_valid,
        y_valid,
    )

    print(f"Validation training time: {validation_training_seconds:.2f} seconds")
    print(f"Validation training time: {validation_training_seconds / 60:.2f} minutes")
    print("Best iteration:", validation_model.best_iteration_)
    print("Feature columns:")
    print(feature_columns)
    print()
    print("Categorical columns:")
    print(categorical_columns)
    print()
    print(f"Validation MAE: {validation_mae:.4f}")

    final_model, final_training_seconds = train_final_model_on_full_data(
        df,
        feature_columns,
        categorical_columns,
    )

    print(f"Final model training time: {final_training_seconds:.2f} seconds")
    print(f"Final model training time: {final_training_seconds / 60:.2f} minutes")

    save_model_artifacts(
        model=final_model,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        artifact_directory=artifact_directory,
        frequency=frequency,
        weather_variant=weather_variant,
    )

    print("Final model saved.")


######################################################################################################

# Use one of the following:

train_lightgbm_model("hourly", "weather_null", True)

# train_lightgbm_model("hourly", "weather_null", True)
# train_lightgbm_model("hourly", "with_weather", True)
# train_lightgbm_model("hourly", "without_weather", False)
# train_lightgbm_model("monthly", "", False)