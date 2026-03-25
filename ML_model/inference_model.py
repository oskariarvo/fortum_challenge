import pandas as pd
import joblib
from pathlib import Path


def build_artifact_directory(frequency: str, weather_variant: str):
    """
    Build the directory path where model artifacts and datasets are stored.

    Args:
        frequency: Dataset frequency, such as `hourly` or `monthly`.
        weather_variant: Weather configuration suffix, such as
            `with_weather`, `without_weather`, or `weather_null`.
            Use an empty string if no variant is used.

    Returns:
        A Path object pointing to the artifact directory.
    """
    if weather_variant:
        return Path(f"{frequency}_{weather_variant}")
    return Path(f"{frequency}")


def load_model_artifacts(
    artifact_dir: Path,
    model_suffix: str
):
    """
    Load trained model and metadata from disk.

    Args:
        artifact_dir: Directory containing saved artifacts.
        model_suffix: Suffix used in file names.

    Returns:
        A tuple containing:
        - trained model
        - feature column list
        - categorical column list
    """
    model = joblib.load(artifact_dir / f"lightgbm_fortum_model_{model_suffix}.pkl")
    feature_cols = joblib.load(artifact_dir / f"feature_cols_{model_suffix}.pkl")
    categorical_cols = joblib.load(artifact_dir / f"categorical_cols_{model_suffix}.pkl")

    return model, feature_cols, categorical_cols


def load_inference_dataset(artifact_dir: Path, frequency: str, weather_variant: str):
    """
    Load inference dataset from CSV.

    Args:
        artifact_dir: Directory containing inference dataset.
        frequency: Dataset frequency.
        weather_variant: Weather configuration suffix.

    Returns:
        A pandas DataFrame containing inference data.
    """
    if weather_variant:
        path = artifact_dir / f"inference_dataset_{frequency}_{weather_variant}.csv"
    else:
        path = artifact_dir / f"inference_dataset_{frequency}.csv"

    return pd.read_csv(path)


def preprocess_inference_data(df: pd.DataFrame, categorical_columns: list[str]):
    """
    Preprocess inference dataset before prediction.

    This includes:
    - parsing `timestamp_utc` to datetime
    - sorting rows
    - casting categorical columns

    Args:
        df: Raw inference DataFrame.
        categorical_columns: Columns to cast as categorical.

    Returns:
        A preprocessed pandas DataFrame.
    """
    processed_df = df.copy()

    processed_df["timestamp_utc"] = pd.to_datetime(
        processed_df["timestamp_utc"],
        utc=True
    )

    processed_df = processed_df.sort_values(
        ["group_id", "timestamp_utc"]
    )

    for col in categorical_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype("category")

    return processed_df


def generate_predictions(model, df: pd.DataFrame, feature_columns: list[str]):
    """
    Generate predictions using a trained model.

    Args:
        model: Trained LightGBM model.
        df: Preprocessed inference DataFrame.
        feature_columns: Feature column names used by the model.

    Returns:
        A DataFrame with predictions added as `target_consumption`.
    """
    prediction_df = df.copy()

    X = prediction_df[feature_columns]
    predictions = model.predict(X)

    prediction_df["target_consumption"] = predictions

    return prediction_df


def save_predictions(df: pd.DataFrame, artifact_dir: Path, frequency: str, weather_variant: str):
    """
    Save predictions to CSV.

    Args:
        df: DataFrame containing predictions.
        artifact_dir: Output directory.
        frequency: Dataset frequency.
        weather_variant: Weather configuration suffix.

    Returns:
        None.
    """
    if weather_variant:
        filename = f"consumption_{frequency}_predictions_{weather_variant}.csv"
    else:
        filename = f"consumption_{frequency}_predictions.csv"

    df.to_csv(artifact_dir / filename, index=False)
    print("Predictions saved.")


def predict_consumption(frequency: str, weather_variant: str):
    """
    Run inference using a trained LightGBM model.

    The workflow:
    - loads model and metadata
    - loads inference dataset
    - preprocesses data
    - generates predictions
    - saves predictions to CSV

    Args:
        frequency: Dataset frequency (`hourly` or `monthly`).
        weather_variant: Weather configuration suffix.
            Use an empty string if not applicable.

    Returns:
        None.
    """
    artifact_dir = build_artifact_directory(frequency, weather_variant)

    if weather_variant:
        model_suffix = f"{frequency}_{weather_variant}"
    else:
        model_suffix = frequency

    model, feature_cols, categorical_cols = load_model_artifacts(
        artifact_dir,
        model_suffix
    )

    df = load_inference_dataset(
        artifact_dir,
        frequency,
        weather_variant
    )

    df = preprocess_inference_data(df, categorical_cols)

    df_with_predictions = generate_predictions(
        model,
        df,
        feature_cols
    )

    save_predictions(
        df_with_predictions,
        artifact_dir,
        frequency,
        weather_variant
    )


######################################################################################################

# Use one of the following:

predict_consumption("hourly", "weather_null")

# predict_consumption("hourly", "weather_null")
# predict_consumption("hourly", "with_weather")
# predict_consumption("hourly", "without_weather")
# predict_consumption("monthly", "")