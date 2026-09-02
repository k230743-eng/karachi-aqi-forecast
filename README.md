# Karachi AQI Predictor

An end-to-end, serverless machine-learning system that forecasts Karachi's **US AQI for the next 72 hours**, keeps its feature data current, retrains itself every day, versions models in Hopsworks, and publishes the latest forecast through a Streamlit dashboard.

**Author:** Ruhab Iqbal  
**Company:** 10 Pearls  
**Submission date:** 2 September 2026

[Live Streamlit App](https://karachi-aqi-forecast-sypfn2an46wrsyxksldyz3.streamlit.app/) | [GitHub Repository](https://github.com/k230743-eng/karachi-aqi-forecast)

---

## What the system does

The project started as a three-day AQI forecasting task and was developed into a complete ML pipeline rather than a one-off notebook. The production path is:

```text
Open-Meteo
    |
    v
Hourly live feature pipeline
    |
    v
Hopsworks Feature Store
    |
    +----------------------+
    |                      |
    v                      v
Latest features      Daily target update
    |                      |
    v                      v
Latest registered    Daily retraining
XGBoost bundle             |
    |                      v
    |               Hopsworks Model Registry
    v
72 hourly predictions
    |
    +--> Hopsworks prediction feature group
    |
    +--> outputs/predictions/latest_72h_forecast.csv
                |
                v
         Streamlit dashboard
```

Two GitHub Actions workflows automate the system:

- **Live AQI Pipeline:** runs hourly, repairs any missing data gap, refreshes recent features, generates a new 72-hour forecast, and publishes the updated forecast CSV.
- **Daily AQI Retraining:** updates mature 1-72 hour targets, retrains all 72 direct XGBoost models, and registers a new versioned model bundle in Hopsworks.

---

## Data

Historical data is collected for Karachi using Open-Meteo APIs at approximately **24.8607 N, 67.0011 E**.

The original backfill covers **1 August 2022 to 30 June 2026**:

- 34,320 hourly air-quality rows
- 34,320 hourly weather rows
- 34,051 final engineered historical rows after cleaning and the 168-hour lookback requirement

Air-quality variables include PM10, PM2.5, carbon monoxide, nitrogen dioxide, sulphur dioxide, ozone, dust, aerosol optical depth, and US AQI. Weather variables include temperature, relative humidity, dew point, surface pressure, precipitation, cloud cover, wind speed, wind direction, and wind gusts.

> The Open-Meteo air-quality feed is model-based rather than a Karachi ground-station feed. This is an important limitation when interpreting the forecasts as a real-world air-quality product.

---

## EDA findings that changed the modelling approach

EDA was used to decide which temporal and historical features were worth engineering instead of removing variables only because of weak linear correlations.

Key findings:

- AQI is **right-skewed**: median about **83**, mean about **92.6**, with occasional severe events and a historical maximum of **297**.
- **PM2.5** has the strongest direct positive relationship with AQI (about **0.725** correlation in the exploratory analysis).
- AQI has a clear **daily cycle**, with the highest average levels around **19:00**.
- Seasonality is strong: winter months, especially **January, November and December**, are substantially worse than late spring; **May** has the lowest monthly average in the historical sample.
- Weekday differences are comparatively small.
- Wind speed is negatively associated with AQI, consistent with pollutant dispersion.
- Humidity has a weaker, scattered relationship; precipitation has little overall linear correlation.
- Relationships are visibly nonlinear, which motivated testing tree-based models rather than relying only on linear regression.

Selected EDA outputs are kept under [`outputs/graphs/`](outputs/graphs/), including:

- [`aqi_over_time.png`](outputs/graphs/aqi_over_time.png)
- [`aqi_distribution.png`](outputs/graphs/aqi_distribution.png)
- [`average_aqi_by_hour.png`](outputs/graphs/average_aqi_by_hour.png)
- [`average_aqi_by_month.png`](outputs/graphs/average_aqi_by_month.png)
- [`correlation_heatmap.png`](outputs/graphs/correlation_heatmap.png)
- [`pm2_5_vs_aqi.png`](outputs/graphs/pm2_5_vs_aqi.png)
- [`wind_speed_vs_aqi.png`](outputs/graphs/wind_speed_vs_aqi.png)

---

## Feature engineering

The final historical feature table contains **72 columns including the timestamp**. The production models use **71 historical/current inputs** plus **14 target-time/future-weather inputs**, for **85 model features** per horizon.

Historical/current features include:

- current pollutant concentrations and AQI
- current weather conditions
- hour, day of week, month, weekend flag
- AQI lags: 1, 3, 6, 12, 24, 48, 72 and 168 hours
- PM2.5 lags: 1, 3, 6, 12, 24, 48 and 72 hours
- PM10 lags: 1, 3, 6, 12, 24, 48 and 72 hours
- rolling means over 3, 6, 12, 24, 48 and 72 hours for AQI, PM2.5 and PM10
- AQI changes over 1, 3, 6, 12, 24, 48 and 72 hours
- AQI rolling standard deviation over 24 and 72 hours

Each horizon also receives weather/time information for the target hour:

- future temperature, humidity, dew point, pressure, precipitation and cloud cover
- future wind speed and wind gusts
- sine/cosine encoding of future wind direction
- target hour, day of week, month and weekend flag

A separate Ridge experiment tested broader cyclical encodings. They did not improve the 72-hour validation result enough to justify replacing the simpler final time representation, so the production feature set retained the stronger empirically supported additions: long lags, rolling summaries, trends, volatility, and horizon-specific future weather.

---

## Models evaluated

The project did not jump directly to XGBoost. Baselines and several model families were compared chronologically.

| Model | 1h MAE | 6h MAE | 24h MAE | 48h MAE | 72h MAE | 72h R² |
|---|---:|---:|---:|---:|---:|---:|
| Persistence baseline | 1.40 | 6.19 | 12.92 | 16.92 | 18.82 | -0.181 |
| Ridge Regression | 1.35 | 4.79 | 10.52 | 14.28 | 15.25 | 0.270 |
| Random Forest | **0.45** | **2.91** | 10.52 | 15.23 | 16.61 | 0.145 |
| TensorFlow dense network | - | - | - | - | 16.37 | 0.137 |
| Direct XGBoost | 1.03 | 3.15 | **10.40** | **13.80** | **14.44** | **0.333** |
| Latest Hopsworks retrain | 0.86 | 2.74 | **9.12** | **11.90** | **12.57** | **0.380** |

Random Forest was extremely strong at very short horizons, while Ridge generalized better than Random Forest farther out. The final direct XGBoost approach provided the best overall long-horizon result and a smooth degradation pattern across the full 1-72 hour forecast range.

The latest Hopsworks retraining results in the repository have an average MAE of approximately **9.54**, average RMSE of **13.35**, and average R² of **0.580** across all 72 horizons. R² remains at or above **0.70 through roughly the first 21 forecast hours**.

Full horizon-by-horizon metrics are available in [`outputs/metrics/xgboost_hourly_hopsworks_1_to_72_metrics.csv`](outputs/metrics/xgboost_hourly_hopsworks_1_to_72_metrics.csv).

---

## Why 72 separate XGBoost models?

The final system uses **direct multi-horizon forecasting**: one model for +1 hour, one for +2 hours, and so on through +72 hours.

This was chosen instead of recursively feeding predictions back into the model because recursive forecasting can compound errors over a three-day window. Each direct model learns the relationship between the current state and one specific future horizon and can use weather information for that target hour directly.

The XGBoost configuration is deliberately regularized (`max_depth=2`, `min_child_weight=20`, subsampling, column sampling, L1/L2 regularization) to reduce overfitting while still capturing nonlinear relationships.

---

## Explainability with SHAP

SHAP explanations were generated at **1h, 6h, 12h, 24h, 48h and 72h**, with global bar plots, beeswarm plots and local waterfall explanations.

The explanations show an intuitive shift in the forecast mechanism as the horizon grows:

- **1 hour:** current AQI dominates, followed by recent PM2.5 and short rolling/trend features.
- **6-24 hours:** PM2.5 rolling history remains important, while future weather begins to contribute more strongly.
- **72 hours:** future pressure and dew point become the two strongest global features in the saved SHAP run; sulphur dioxide, month, longer PM2.5 summaries and wind-related features also become important.

This horizon-dependent change is useful evidence that the longer-horizon models are not merely copying the current AQI.

All SHAP outputs are in [`outputs/shap/`](outputs/shap/).

---

## Hopsworks feature and model management

The production ML state is stored in Hopsworks rather than local files alone.

### Feature groups

- `karachi_aqi_features` v2 - historical/live pollutants, weather and engineered features
- `karachi_aqi_targets` v1 - delayed labels `aqi_target_1h` through `aqi_target_72h`
- `karachi_aqi_predictions` v1 - persisted live forecast rows

The target updater only creates a complete training row once its full +72h outcome is known. This avoids pretending that future labels are available before they actually occur.

### Model Registry

Each daily retraining creates a new version of `karachi_aqi_xgboost_72h`. The registered artifact is a single ZIP containing:

- 72 `joblib` XGBoost models
- `feature_columns.json`
- `metrics.csv`
- `metadata.json`

The live predictor queries the registry and automatically selects the newest model version.

---

## Self-healing hourly feature pipeline

`13_live_feature_pipeline.py` was designed to recover from missed executions instead of assuming every previous hourly job succeeded.

On every run it:

1. checks the newest timestamp already stored in Hopsworks;
2. determines the latest fully completed Karachi hour;
3. calculates the first missing hour;
4. also refreshes the most recent **48 hours** in case recent source data has been revised;
5. fetches a **192-hour lookback** so the longest 168-hour lag can be reconstructed safely;
6. downloads in 30-day chunks with HTTP retry/backoff for rate limits and transient server errors;
7. verifies there are no hourly gaps before using row-based lag operations;
8. regenerates the exact feature schema and upserts the required rows into Hopsworks.

If the workflow stops for several hours or days, the next run therefore catches up automatically rather than permanently leaving holes in the feature history.

---

## Automated retraining

`16_update_targets.py` handles delayed labels. If the newest feature timestamp is `t`, the newest fully mature training timestamp is `t - 72 hours`. For each eligible row it creates all 72 target columns from the later observed AQI values.

The daily workflow then:

```text
update mature targets
        ->
read latest feature + target groups
        ->
chronological 80/20 split
        ->
train 72 XGBoost models
        ->
evaluate MAE / RMSE / R²
        ->
register a new model version
```

This means daily retraining genuinely incorporates newly accumulated data rather than repeatedly training on a frozen historical snapshot.

---

## Live prediction and dashboard

`14_predict_live.py`:

- reads the newest engineered historical feature row;
- retrieves the newest registered model version;
- downloads the 72-model bundle when required;
- fetches future weather from Open-Meteo;
- builds the exact horizon-specific model inputs;
- generates all 72 hourly AQI predictions;
- writes the forecast to Hopsworks and to `outputs/predictions/latest_72h_forecast.csv`.

The deployed Streamlit app reads the latest published CSV and shows:

- next-hour AQI
- 72-hour average, minimum and maximum
- AQI health category and alert message
- 72-hour AQI curve
- key horizons (1, 6, 12, 24, 48, 72 hours)
- temperature, humidity, precipitation and wind forecasts
- complete hourly forecast table

The dashboard is intentionally isolated from the Hopsworks Python environment because the current Streamlit and Hopsworks stacks require incompatible protobuf ranges. Keeping the dashboard in its own lightweight environment avoids a fragile deployment dependency conflict.

**Live application:** https://karachi-aqi-forecast-sypfn2an46wrsyxksldyz3.streamlit.app/

---

## Repository structure

```text
karachi-aqi-forecast/
├── .github/
│   └── workflows/
│       ├── live_aqi_pipeline.yml
│       └── daily_retraining.yml
├── data/
├── experiments/
├── models/
├── outputs/
│   ├── graphs/
│   ├── metrics/
│   ├── predictions/
│   └── shap/
├── 1_collect_data.py
├── 2_prepare_data.py
├── 3_train_models.py
├── 4_train_tensorflow.py
├── 5_train_hourly_xgboost.py
├── 6_shap_explanations.py
├── 7_upload_features_to_hopsworks.py
├── 8_upload_targets_to_hopsworks.py
├── 9_create_feature_view.py
├── 10_train_from_hopsworks.py
├── 11_train_final_models_from_hopsworks.py
├── 12_register_models_hopsworks.py
├── 13_live_feature_pipeline.py
├── 14_predict_live.py
├── 15_dashboard.py
├── 16_update_targets.py
├── eda.py
├── requirements-pipeline.txt
├── requirements-live.txt
├── requirements-dashboard.txt
└── requirements.txt
```

---

## Running locally

### 1. Clone the repository

```bash
git clone https://github.com/k230743-eng/karachi-aqi-forecast.git
cd karachi-aqi-forecast
```

### 2. Pipeline environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-live.txt
```

For the full experiment environment, use `requirements-pipeline.txt`.

### 3. Dashboard environment

The dashboard is kept separate from Hopsworks because of the protobuf dependency conflict:

```powershell
python -m venv .dashboard-venv
.\.dashboard-venv\Scripts\Activate.ps1
pip install -r requirements-dashboard.txt
streamlit run 15_dashboard.py
```

---

## GitHub Actions secrets

The automated workflows expect these repository secrets:

```text
HOPSWORKS_API_KEY
HOPSWORKS_HOST
HOPSWORKS_PROJECT
```

Secrets are never stored in the repository. Local execution uses the local Hopsworks certificate folder, while GitHub Actions authenticates through the API key.

---

## Main design decisions

A few decisions had an outsized effect on the final system:

1. **Keep extreme AQI events unless there is evidence they are erroneous.** They are rare, but they are exactly the conditions an AQI forecaster should learn to handle.
2. **Use chronological evaluation.** Random train/test splits would leak future behavior into the past and overstate forecasting accuracy.
3. **Do not remove weakly correlated weather variables only from a heatmap.** Tree models can exploit nonlinear interactions that linear correlation misses.
4. **Use direct multi-horizon models.** This avoids recursive error accumulation over a 72-hour window.
5. **Treat targets as delayed data.** A +72h label is only written after that hour has occurred.
6. **Make live ingestion repair itself.** Missing runs should create a temporary delay, not a permanent hole in the dataset.
7. **Version models and always load the newest registry version.** Daily retraining is only useful if production predictions actually move to the new model.

---

## Limitations

- Historical air-quality values come from the selected Open-Meteo air-quality feed, not a dedicated Karachi regulatory ground-station network.
- Offline training uses historical weather observed at the target hour, while production uses weather forecasts for that hour. This creates some train/serve covariate mismatch and makes offline results somewhat optimistic.
- Forecast accuracy naturally falls as the horizon increases; the 72-hour model remains useful but is materially less certain than the first several hours.
- Rare severe-pollution episodes are under-represented compared with moderate AQI conditions.
- The Streamlit deployment consumes a forecast CSV committed by the hourly workflow. This is intentionally simple and serverless for the project, but a production service could expose forecasts through a dedicated API or object store instead.
- No formal prediction interval or probabilistic uncertainty band is currently shown.

---

## Future work

- retrain using archived **weather forecasts** rather than realized historical weather to better match production inputs;
- add uncertainty intervals and explicit confidence information;
- evaluate performance separately for unhealthy and hazardous AQI episodes;
- add model-promotion rules so a new daily model is promoted only when validation metrics pass defined thresholds;
- incorporate additional or ground-station data sources when available;
- extend the pipeline to multiple cities;
- add drift monitoring for both feature distributions and forecast errors.

---

## Submission coverage

The project implements the requirements in the original Pearls AQI Predictor brief:

- historical feature/target backfill
- external weather and pollution data collection
- EDA and trend analysis
- time, lag, rolling, change and weather feature engineering
- Hopsworks Feature Store
- baselines, Scikit-learn models, deep learning and gradient boosting experiments
- MAE, RMSE and R² evaluation
- SHAP explanations
- Hopsworks Model Registry
- hourly automated feature/prediction pipeline
- daily automated target update/retraining pipeline
- hazardous AQI alert logic
- public interactive Streamlit dashboard
- end-to-end serverless deployment

---

## Author

**Ruhab Iqbal**  
10 Pearls - AQI Forecasting Project  
2 September 2026
