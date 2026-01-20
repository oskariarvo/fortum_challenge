import requests
from datetime import date
import json
from pyspark.sql import SparkSession

def fetch_calendar_data(start_date: str, end_date: str):
    """
    Fetches Finnish calendar and holiday data for a given date range. 
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
    

def write_raw(data, destination):
    """
    Creates a spark session, creates a dataframe from the data and writes it to a delta table.
    """
    spark = SparkSession.builder.getOrCreate()
    rows = [{"timestamp": ts, **vals} for ts, vals in data.items()]
    print("Created rows.")
    df = spark.createDataFrame(rows)
    print("Created df.")
    df.write.format("delta").mode("overwrite").saveAsTable(destination)
    print("Wrote to table.")


if __name__ == "__main__":
    # Define a date range for the request
    start = date(2021, 1, 1)
    end = date(2024, 9, 30)
    destination = "fortum_challenge_data.01_bronze.calendar_data_raw"

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print(f"Fetching calendar data from {start_str} to {end_str}...")
    calendar_data = fetch_calendar_data(start_str, end_str)

    if calendar_data:
        print("Successfully fetched data")
        write_raw(calendar_data, destination)
