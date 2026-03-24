import requests
from pyspark.sql import SparkSession
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, date, timedelta
import os
import json
import sys

def fetch_weather_data_for_city(start, end, city):
    """
    Fetch hourly weather observations for a single Finnish city from the FMI Open Data API.

    The function requests hourly weather measurements for the given UTC time range
    and parses the XML response into a dictionary keyed by timestamp. Each timestamp
    contains the city name and the available weather parameter values returned by
    the API.

    Requested parameters:
        - TA_PT1H_AVG: hourly average temperature
        - WS_PT1H_AVG: hourly average wind speed
        - PRA_PT1H_ACC: hourly accumulated precipitation

    Args:
        start: Start time in UTC ISO 8601 format, for example
            "2024-09-21T00:00:00Z".
        end: End time in UTC ISO 8601 format, for example
            "2024-09-30T00:00:00Z".
        city: Name of the city/place as expected by the FMI API, for example
            "helsinki".

    Returns:
        A dictionary where each key is a timestamp string and each value is a
        dictionary containing the city and weather parameter values for that
        timestamp.

        Example:
            {
                "2024-09-21T00:00:00Z": {
                    "city": "helsinki",
                    "TA_PT1H_AVG": "12.3",
                    "WS_PT1H_AVG": "4.5",
                    "PRA_PT1H_ACC": "0.0"
                }
            }

    Side Effects:
        Prints progress and error messages to standard output.

    Raises:
        No Python exceptions are propagated to the caller for request or XML
        parsing errors. On failure, the script prints an error message and
        exits with status code 1 using sys.exit(1).
    """

    url = "https://opendata.fmi.fi/wfs"

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "storedquery_id": "fmi::observations::weather::hourly::timevaluepair",

        # Optional parameters:
        "parameters": "TA_PT1H_AVG,WS_PT1H_AVG,PRA_PT1H_ACC",  # temperature + wind + precipitation
        "starttime": start,   # 2024-10-01T00:00:00Z
        "endtime": end,       # 2024-10-03T00:00:00Z
        "timestep": "60",     # 60-minute interval
        "place": city         # e.g. "helsinki" 
    }

    print(f"Fetching data for {city} from {start} to {end}.")

    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.exceptions.RequestException as e:
        print(json.dumps({"message": str(e), "severity": "ERROR"}))
        sys.exit(1)

    if response.status_code != 200:
        print(json.dumps({
            "message": f"HTTP {response.status_code}: {response.text}",
            "severity": "ERROR"
        }))
        print("Fetch was not successful.")
        sys.exit(1)

    print("Fetched data successfully.")


    # Parsing XML
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        print(json.dumps({"message": "Invalid XML from FMI", "severity": "ERROR"}))
        sys.exit(1)

    ns = {
        'wfs': 'http://www.opengis.net/wfs/2.0',
        'om': 'http://www.opengis.net/om/2.0',
        'gml': 'http://www.opengis.net/gml/3.2',
        'wml2': 'http://www.opengis.net/waterml/2.0',
    }

    data = {}

    for measurement in root.findall(".//wml2:MeasurementTimeseries", ns):
        raw_id = measurement.attrib.get("{http://www.opengis.net/gml/3.2}id", "unknown")
        param_name = raw_id.split('-')[-1]

        for point in measurement.findall(".//wml2:MeasurementTVP", ns):
            time = point.find("wml2:time", ns).text
            value = point.find("wml2:value", ns).text

            if time not in data:
                data[time] = {
                "city": city
            }

            data[time][param_name] = value

    print("Created data.")
    return data




def fetch_weather_data_for_cities(start_utc, end_utc, cities):
    """
    Fetch hourly weather observations for multiple cities and combine them into row-based data.

    The function calls fetch_weather_data_for_city() separately for each city, transforms the
    timestamp-keyed dictionaries into row dictionaries, and concatenates all rows
    into a single list suitable for Spark DataFrame creation.

    Args:
        start_utc: Start time in UTC ISO 8601 format.
        end_utc: End time in UTC ISO 8601 format.
        cities: List of city names to fetch from the FMI API.

    Returns:
        A list of row dictionaries. Each row contains:
            - timestamp
            - city
            - one or more weather parameter fields returned by the API

    Side Effects:
        Makes one API request per city and prints progress messages.

    Raises:
        Propagates termination behavior from fetch_weather_data_for_city(). If one city fetch
        fails, the script exits.
    """
    all_rows = []
    for city in cities:
        data = fetch_weather_data_for_city(start_utc, end_utc, city)
        rows = [{"timestamp": ts, **vals} for ts, vals in data.items()]
        all_rows.extend(rows)
    print("Created rows.")
    return all_rows


def write_weather_to_bronze_table(data, destination):
    """
    Write weather observation data to a Delta table.

    The function creates a Spark DataFrame from a list of row dictionaries and
    writes the result to the given destination table in Delta format using
    overwrite mode.

    Args:
        data: A list of row dictionaries, typically returned by
            fetch_weather_data_for_cities().
        destination: Fully qualified Spark table name to write to.

    Returns:
        None.

    Side Effects:
        - Creates a Spark session if one does not already exist.
        - Overwrites any existing data in the destination Delta table.
    """
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data)
    print("Created df.")
    df.write.format("delta").mode("overwrite").saveAsTable(destination)
    print("Wrote to table.")

if __name__ == "__main__":
    # Fetch hourly weather data for the inference period and write it to bronze storage.
    # Data from 21.9.2024 to 29.9.2024 (and first hour of 30.9.2024)
    start = date(2024, 9, 21) 
    end = date(2024, 9, 30) 

    start_utc = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Destination catalog.schema.table
    destination = "fortum_challenge_data.01_bronze.weather_hourly_inference_raw"

    # Cities we want data from
    cities = ["helsinki", "turku", "tampere", "vaasa", "kuopio", "jyväskylä", "oulu", "rovaniemi", "lahti", "lappeenranta", "joensuu", "hämeenlinna", "espoo", "seinäjoki", "vantaa", "pori"]

    data = fetch_weather_data_for_cities(start_utc, end_utc, cities)
    write_weather_to_bronze_table(data, destination)