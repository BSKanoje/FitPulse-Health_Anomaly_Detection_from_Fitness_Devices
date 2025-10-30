# import pandas as pd
# import numpy as np
# from prophet import Prophet
# from tsfresh import extract_features
# from tsfresh.utilities.dataframe_functions import impute
# import warnings
# warnings.filterwarnings('ignore')

# def extract_features_tsfresh(df, metric):
#     """Extract TSFresh features for one metric (univariate, fixes API error)."""
#     try:
#         # Prepare DF: reset index, add ID, rename to 'time'
#         df_metric = df[['timestamp', metric]].reset_index(drop=True)
#         df_metric['id'] = 1  # Single ID
#         df_metric = df_metric.rename(columns={'timestamp': 'time'})
#         df_metric = df_metric[['id', 'time', metric]]  # Clean columns
        
#         # Extract features (correct args: no column_value!)
#         features = extract_features(
#             df_metric,
#             column_id='id',
#             column_sort='time',
#             column_kind=metric,  # Specify the kind (metric name)
#             n_jobs=1
#         )
#         if not features.empty:
#             impute(features)
#             return features.iloc[0].to_dict()
#         else:
#             raise ValueError("Empty features")
#     except Exception as e:
#         print(f"TSFresh fallback for {metric}: {e}")
#         # Basic features (always works)
#         series = df[metric].dropna()
#         if len(series) == 0:
#             return {}
#         return {
#             f'{metric}_mean': series.mean(),
#             f'{metric}_std': series.std(),
#             f'{metric}_min': series.min(),
#             f'{metric}_max': series.max(),
#             f'{metric}_count': len(series),
#             f'{metric}_variance': series.var()
#         }

# def extract_all_features(df):
#     """Extract for all metrics. FIXED: Proper DataFrame from list of dicts."""
#     all_features_dicts = []  # List of dicts instead of Series
#     for metric in ['heart_rate', 'steps', 'sleep_duration']:
#         feats = extract_features_tsfresh(df, metric)
#         feats['metric'] = metric  # Add metric as key in dict
#         all_features_dicts.append(feats)
    
#     if all_features_dicts:
#         features_df = pd.DataFrame(all_features_dicts).set_index('metric')
#         return features_df
#     return pd.DataFrame()  # Empty if no features

# def model_with_prophet(df, metric):
#     """Prophet model with correct timestamp alignment."""
#     try:
#         # ✅ Ensure timestamp column exists
#         if 'timestamp' not in df.columns:
#             raise ValueError("Missing 'timestamp' column")

#         # ✅ Prepare data for Prophet
#         prophet_df = df[['timestamp', metric]].rename(columns={'timestamp': 'ds', metric: 'y'}).dropna()

#         if len(prophet_df) < 10:
#             raise ValueError("Insufficient data for Prophet")

#         # ✅ Fit Prophet model
#         model = Prophet(daily_seasonality=True, weekly_seasonality=False)
#         model.fit(prophet_df)

#         # ✅ Predict same range as actual (no future forecast yet)
#         future = prophet_df[['ds']]
#         forecast = model.predict(future)

#         # ✅ Calculate residuals (actual - predicted)
#         forecast = forecast[['ds', 'yhat']]
#         prophet_df = prophet_df.set_index('ds')
#         forecast = forecast.set_index('ds')

#         common_idx = prophet_df.index.intersection(forecast.index)
#         residuals = prophet_df.loc[common_idx, 'y'] - forecast.loc[common_idx, 'yhat']

#         return model, forecast.reset_index(), residuals

import pandas as pd
import numpy as np
from prophet import Prophet
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import warnings
warnings.filterwarnings('ignore')

def extract_features_tsfresh(df, metric):
    """Extract TSFresh features for one metric (univariate, fixes API error)."""
    try:
        # Prepare DF: reset index, add ID, rename to 'time'
        df_metric = df[['timestamp', metric]].reset_index(drop=True)
        df_metric['id'] = 1  # Single ID
        df_metric = df_metric.rename(columns={'timestamp': 'time'})
        df_metric = df_metric[['id', 'time', metric]]  # Clean columns
        
        # Extract features (correct args: no column_value!)
        features = extract_features(
            df_metric,
            column_id='id',
            column_sort='time',
            column_kind=metric,  # Specify the kind (metric name)
            n_jobs=1
        )
        if not features.empty:
            impute(features)
            return features.iloc[0].to_dict()
        else:
            raise ValueError("Empty features")
    except Exception as e:
        print(f"TSFresh fallback for {metric}: {e}")
        # Basic features (always works)
        series = df[metric].dropna()
        if len(series) == 0:
            return {}
        return {
            f'{metric}_mean': series.mean(),
            f'{metric}_std': series.std(),
            f'{metric}_min': series.min(),
            f'{metric}_max': series.max(),
            f'{metric}_count': len(series),
            f'{metric}_variance': series.var()
        }

def extract_all_features(df):
    """Extract for all metrics. FIXED: Proper DataFrame from list of dicts."""
    all_features_dicts = []  # List of dicts instead of Series
    for metric in ['heart_rate', 'steps', 'sleep_duration']:
        feats = extract_features_tsfresh(df, metric)
        feats['metric'] = metric  # Add metric as key in dict
        all_features_dicts.append(feats)
    
    if all_features_dicts:
        features_df = pd.DataFrame(all_features_dicts).set_index('metric')
        return features_df
    return pd.DataFrame()  # Empty if no features

def model_with_prophet(df, metric):
    """Prophet model with correct timestamp alignment."""
    try:
        # ✅ Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            raise ValueError("Missing 'timestamp' column")

        # ✅ Prepare data for Prophet
        prophet_df = df[['timestamp', metric]].rename(columns={'timestamp': 'ds', metric: 'y'}).dropna()

        if len(prophet_df) < 10:
            raise ValueError("Insufficient data for Prophet")

        # ✅ Fit Prophet model
        model = Prophet(daily_seasonality=True, weekly_seasonality=False)
        model.fit(prophet_df)

        # ✅ Predict same range as actual (no future forecast yet)
        future = prophet_df[['ds']]
        forecast = model.predict(future)

        # ✅ Calculate residuals (actual - predicted)
        forecast = forecast[['ds', 'yhat']]
        prophet_df = prophet_df.set_index('ds')
        forecast = forecast.set_index('ds')

        common_idx = prophet_df.index.intersection(forecast.index)
        residuals = prophet_df.loc[common_idx, 'y'] - forecast.loc[common_idx, 'yhat']

        return model, forecast.reset_index(), residuals

    except Exception as e:
        print(f"Prophet fallback for {metric}: {e}")
        # Fallback: moving average if Prophet fails
        series = df[metric]
        window = max(1, len(series) // 10)
        ma = series.rolling(window=window).mean()
        residuals = series - ma.fillna(series.mean())
        return None, None, residuals
