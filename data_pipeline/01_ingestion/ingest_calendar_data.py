import requests
from datetime import date
import json
from pyspark.sql import SparkSession

def fetch_calendar_data(start_date: str, end_date: str):
    """
    Fetch Finnish calendar and holiday data from the Holiday Calendar API.

    Args:
        start_date: Start of the requested date range in YYYY-MM-DD format.
        end_date: End of the requested date range in YYYY-MM-DD format.

    Returns:
        A dictionary parsed from the API JSON response, where each key is a
        date/timestamp and each value contains calendar-related fields for that
        day. Returns None if the request fails.

    Raises:
        No exception is propagated from the request layer; request-related
        errors are caught and logged, and the function returns None instead.
    """
    base_url = "https://holiday-calendar.fi/api/calendar"
    params = {
        "start": start_date,
        "end": end_date
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the API request: {e}")
        return None
    

def write_calendar_to_bronze_table(data, destination):
    """
    Write calendar data to a Delta table.

    The input data is expected to be a dictionary where keys are timestamps
    and values are dictionaries of calendar attributes. The function flattens
    this structure into Spark rows, creates a DataFrame, and overwrites the
    destination Delta table.

    Args:
        data: Calendar data returned by fetch_calendar_data().
        destination: Fully qualified Spark table name to write to.

    Returns:
        None.

    Side Effects:
        - Creates a Spark session if one does not already exist.
        - Overwrites any existing data in the destination Delta table.
    """
    spark = SparkSession.builder.getOrCreate()
    rows = [{"timestamp": ts, **vals} for ts, vals in data.items()]
    print("Created rows.")
    df = spark.createDataFrame(rows)
    print("Created df.")
    df.write.format("delta").mode("overwrite").saveAsTable(destination)
    print("Wrote to table.")


def fetch_and_store_calendar_data(start: date, end: date, destination: str):
    """
    Fetch calendar data for a given date range and write it to a bronze table.

    The function converts date inputs to the required API format, fetches
    calendar data using the Holiday Calendar API, and writes the result to
    the specified bronze-layer table.

    Args:
        start: Start date of the requested range.
        end: End date of the requested range.
        destination: Fully qualified bronze table name to write to.

    Returns:
        None.

    Side Effects:
        - Makes an API request to fetch calendar data.
        - Writes data to a Delta table in the bronze layer.
        - Prints progress information to standard output.
    """
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print(f"Fetching calendar data from {start_str} to {end_str}...")
    calendar_data = fetch_calendar_data(start_str, end_str)

    if calendar_data:
        print("Successfully fetched data")
        write_calendar_to_bronze_table(calendar_data, destination)


if __name__ == "__main__":

    # Fetch calendar data for the training and hourly inference period.
    fetch_and_store_calendar_data(
        start=date(2021, 1, 1),
        end=date(2024, 9, 30),
        destination="fortum_challenge_data.01_bronze.calendar_data_raw"
    )

    # Fetch calendar data for the monthly inference period.
    fetch_and_store_calendar_data(
        start=date(2024, 10, 1),
        end=date(2025, 9, 30),
        destination="fortum_challenge_data.01_bronze.calendar_monthly_inference_raw"
    )