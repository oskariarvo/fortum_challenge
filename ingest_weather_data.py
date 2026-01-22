
import requests
from pyspark.sql import SparkSession
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, date, timedelta
import os
import json
import sys

def fetch_api_data(start, end, city):
    """
    Fetches Finnish weather data for a given date range. 
    """

    url = "https://opendata.fmi.fi/wfs"

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "storedquery_id": "fmi::observations::weather::hourly::timevaluepair",

        # TA_PT1H_AVG: temperature (lämpötila)
        # PRA_PT1H_ACC: precipitation amount (sademäärä)
        # WS_PT1H_AVG: wind speed (tuulen nopeus)

        # Optional parameters:
        "parameters": "TA_PT1H_AVG,WS_PT1H_AVG,PRA_PT1H_ACC",  # temperature + wind + precipitation
        "starttime": start,   # 2024-10-01T00:00:00Z
        "endtime": end,     # 2024-10-03T00:00:00Z
        "timestep": "60",               # 60-minute interval
        "place": city     # e.g. "helsinki" 
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




def fetch_multiple_cities(start_utc, end_utc, cities):
    """
    Fetches Finnish weather data from all given cities and concatenates the results for a given date range. 
    """
    all_rows = []
    for city in cities:
        data = fetch_api_data(start_utc, end_utc, city)
        rows = [{"timestamp": ts, **vals} for ts, vals in data.items()]
        all_rows.extend(rows)
    print("Created rows.")
    return all_rows


def write_raw(data, destination):
    """
    Creates a spark session, creates a dataframe from the data and writes it to a delta table.
    """
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data)
    print("Created df.")
    df.write.format("delta").mode("overwrite").saveAsTable(destination)
    print("Wrote to table.")


# Data from 01.10.2024 to 03.10.2024
start = date(2024, 10, 1) 
end = date(2024, 10, 3) 

start_utc = start.strftime("%Y-%m-%dT%H:%M:%SZ")
end_utc = end.strftime("%Y-%m-%dT%H:%M:%SZ")

# Destination catalog.schema.table
destination = "fortum_challenge_data.01_bronze.weather_inference_raw"

# Cities we want data from
cities = ["helsinki", "turku", "tampere", "vaasa", "kuopio", "jyväskylä", "oulu", "rovaniemi", "lahti", "lappeenranta", "joensuu"]

data = fetch_multiple_cities(start_utc, end_utc, cities)
write_raw(data, destination)