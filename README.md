# Fortum Energy Consumption Forecasting – Junction 2025 Hackathon

This repository contains my solution for the Fortum Challenge at the Junction 2025 Hackathon, where the goal was to forecast electricity consumption for multiple customer groups in Finland.

The project combines a structured data engineering pipeline with machine learning to generate both short-term (48-hour) and long-term (12-month) forecasts.

---

## 📌 Problem Overview

The task was to predict electricity consumption for 112 customer groups using historical consumption and price data.

Two forecasting horizons were required:

* **Short-term (48-hour, hourly)** → operational energy trading decisions
* **Long-term (12-month, monthly)** → hedging and strategic planning

The objective was to outperform a baseline model using **Forecast Value Added (FVA)**.

---

## 🏗️ Project Structure

```
fortum_challenge/
│
├── data/
│   ├── raw/
│   ├── forecasts/
│
├── data_pipeline/
│   ├── 01_ingestion/
│   ├── 02_cleaning/
│   ├── 03_aggregation/
│   ├── export/
│
├── ML_model/
│   ├── training_model.py
│   ├── inference_model.py
│
├── submission_processing/
│   ├── export_deliverables/
│   ├── submission_formatting/
│   ├── evaluation_hourly.ipynb
│
└── README.md
└── .gitignore
└── requirements.txt
└── fortum_challenge_guidebook.pdf
```

---

### 📁 Data Folder

- `raw/` contains the original (anonymized) dataset provided in the challenge (consumption, prices, etc.)
- `forecasts/` contains the final prediction outputs submitted for evaluation:
  - 48-hour hourly forecasts
  - 12-month monthly forecasts

---

## ⚙️ Data Pipeline

The project follows a layered data engineering approach implemented in Databricks:

1. **Ingestion**

   * Raw consumption, price, weather, and calendar data

2. **Cleaning**

   * Data type normalization
   * Timestamp alignment
   * Handling missing values

3. **Aggregation**

   * Feature-ready datasets for modeling
   * Group-level joins (consumption + price + weather + metadata)

This pipeline produces training and inference datasets exported as CSV files for local model training.

---

## 🤖 Modeling Approach

A LightGBM model was selected due to its strong performance on tabular data and its ability to handle:

- Non-linear relationships  
- Mixed feature types (numerical and categorical)  
- Large-scale datasets efficiently

The same modeling approach was applied for both hourly (48-hour) and monthly (12-month) forecasting tasks, allowing a consistent and scalable pipeline design.

The focus of the solution was placed on feature engineering rather than model complexity, as the problem involves strong temporal patterns and limited future information.

### Key characteristics:

* Single unified modeling approach for both time horizons
* Time-aware sorting (`group_id`, `timestamp_utc`)
* Categorical features included (e.g., group metadata)

### Feature Engineering

The model relies heavily on **lag-based features** due to limited future information:

* Lag features:

  * `lag_24`, `lag_48`, `lag_168`
* Rolling statistics on target:

  * Rolling mean
  * Rolling standard deviation

This approach ensures that inference can be performed without requiring unknown future values.

### External Data

* Finnish public holiday data
* Weather data from public sources

### Model Training & Validation

The model was trained using a time-based validation strategy to reflect real-world forecasting conditions.

- Data was sorted by `group_id` and `timestamp_utc`
- A fixed temporal cutoff was used to split training and validation data:
  - Hourly: 2024-09-15
  - Monthly: 2023-10-01

All observations before the cutoff were used for training, and observations at or after the cutoff were used for validation.

This approach ensures:
- No future data leakage
- Realistic evaluation aligned with forecasting tasks

For additional evaluation, the final 48 hours of historical data were used to simulate the competition prediction window and compare predictions against actual values.

Model performance was evaluated using:
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Forecast Value Added (FVA) relative to baseline

---

## 📊 Results & Evaluation

Three model variants were tested:

| Model           | MAE    | MAPE (%) | FVA (%)   |
| --------------- | ------ | -------- | --------- |
| Weather (null)  | 0.0846 | 7.45     | **+2.21** |
| With weather    | 0.0827 | 7.90     | -3.67     |
| Without weather | 0.0998 | 8.22     | -7.85     |

### Key Findings

* Adding weather features improved performance compared to no weather
* However, the best results were achieved when:

  * Weather features were used during training
  * But set to **null during inference**

This resulted in a **positive FVA (+2.21%)**, outperforming the baseline.

This behavior suggests that:

* The model benefited from learning weather-related patterns
* But may have struggled with generalizing future weather inputs

---

## 🔄 Pipeline Execution

To reproduce the workflow:

1. Run the **data pipeline in Databricks**

   * Execute ingestion → cleaning → aggregation layers in order

2. Export datasets locally

   * Training and inference datasets as CSV files

3. Train the model locally

   * Run `training_model.py`

4. Run inference

   * Run `inference_model.py`

5. Format submission

   * Upload predictions back to Databricks
   * Run `submission_formatting`

6. Evaluate results (optional)

   * Run evaluation scripts in `submission_processing`
   * Note: Evaluation is done on historical data (last 48h of training)

---

## 📦 Data Format

Expected local folder structure:

```
{frequency}_{weather_variant}/
└── training_dataset_{frequency}_{weather_variant}.csv
```

Example:

```
hourly_weather_null/
└── training_dataset_hourly_weather_null.csv
```

---

## 🎯 Key Focus Areas

This project emphasizes:

* **Data Engineering**

  * Structured medallion-style pipeline
  * Clean separation of ingestion, transformation, and modeling

* **Feature Engineering**

  * Robust lag-based design for real-world forecasting constraints
  * Handling missing future exogenous variables

---

## 🧠 Learnings

* Forecasting under limited future information requires careful feature design
* Lag features can outperform naive use of external variables
* More features do not always improve performance
* Real-world constraints (e.g., missing future data) strongly shape model design

---

## 🛠️ Tech Stack

* Python (Pandas, LightGBM, Scikit-learn)
* Databricks (SQL, PySpark)
* Git & GitHub

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

* Intermediate datasets are generated via the data pipeline and are not stored in the repository  
* See `fortum_challenge_guidebook.pdf` for the full challenge description  

---

## 🚀 Summary

This project demonstrates how combining a structured data pipeline with carefully designed features can produce meaningful improvements over baseline forecasts in a real-world energy context.

Despite relatively simple modeling techniques, feature engineering and pipeline design played a key role in achieving positive performance.
