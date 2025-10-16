import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from prophet import Prophet
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Custom Color Palette
COLORS = {
    'primary': '#6366F1',
    'secondary': '#EC4899',
    'accent': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'dark': '#1F2937',
    'light': '#F3F4F6'
}

class TSFreshFeatureExtractor:
    """Extract time-series features using TSFresh"""
    
    def __init__(self):
        self.feature_matrix = None
        self.feature_names = []
        
    def extract_features(self, df: pd.DataFrame, data_type: str, window_size: int = 60):
        report = {'data_type': data_type, 'success': False}
        try:
            df_prepared = self._prepare_data(df, data_type, window_size)
            if df_prepared is None:
                return pd.DataFrame(), report
            
            feature_matrix = extract_features(
                df_prepared,
                column_id='window_id',
                column_sort='timestamp',
                default_fc_parameters=MinimalFCParameters(),
                disable_progressbar=True,
                n_jobs=1
            )
            feature_matrix = impute(feature_matrix)
            feature_matrix = self._remove_constant_features(feature_matrix)
            
            self.feature_matrix = feature_matrix
            self.feature_names = list(feature_matrix.columns)
            
            report.update({
                'features_extracted': len(self.feature_names),
                'feature_windows': len(feature_matrix),
                'success': True
            })
            
            return feature_matrix, report
        except Exception as e:
            report['error'] = str(e)
            return pd.DataFrame(), report
    
    def _prepare_data(self, df: pd.DataFrame, data_type: str, window_size: int):
        """Prepare data for TSFresh dynamically for any numeric column"""
        metric_col = data_type
        if metric_col not in df.columns or not np.issubdtype(df[metric_col].dtype, np.number):
            return None
        
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        prepared_data = []
        step_size = window_size // 2
        
        for i in range(0, len(df_sorted) - window_size + 1, step_size):
            window_data = df_sorted.iloc[i:i+window_size].copy()
            window_data['window_id'] = i // step_size
            prepared_data.append(window_data[['window_id', 'timestamp', metric_col]])
        
        if not prepared_data:
            return None
        
        df_prepared = pd.concat(prepared_data, ignore_index=True)
        df_prepared = df_prepared.rename(columns={metric_col: 'value'})
        return df_prepared
    
    def _remove_constant_features(self, df: pd.DataFrame):
        constant_features = [col for col in df.columns if df[col].std() == 0]
        return df.drop(columns=constant_features) if constant_features else df
    
    def get_top_features(self, n: int = 10):
        if self.feature_matrix is None or self.feature_matrix.empty:
            return pd.DataFrame()
        feature_variance = self.feature_matrix.var().sort_values(ascending=False).head(n)
        return pd.DataFrame({
            'Feature': feature_variance.index,
            'Variance': feature_variance.values,
            'Mean': [self.feature_matrix[f].mean() for f in feature_variance.index]
        })

class ProphetTrendModeler:
    """Model time-series trends using Prophet"""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.residuals = {}
    
    def fit_and_predict(self, df: pd.DataFrame, data_type: str, forecast_periods: int = 100):
        report = {'data_type': data_type, 'success': False}
        try:
            metric_col = data_type
            if metric_col not in df.columns or not np.issubdtype(df[metric_col].dtype, np.number):
                report['error'] = f"Column '{metric_col}' not found or not numeric"
                return pd.DataFrame(), report
            
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            prophet_df = pd.DataFrame({
                'ds': df['timestamp'],
                'y': df[metric_col]
            }).dropna()
            
            prophet_df = prophet_df.groupby('ds').mean().reset_index()
            if len(prophet_df) < 2:
                return pd.DataFrame(), report
            
            model = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False,
                            changepoint_prior_scale=0.05, interval_width=0.95)
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=forecast_periods, freq='min')
            forecast = model.predict(future)
            
            merged = prophet_df.merge(forecast[['ds','yhat','yhat_lower','yhat_upper']], on='ds', how='left')
            merged['residual'] = merged['y'] - merged['yhat']
            merged['residual_abs'] = np.abs(merged['residual'])
            
            self.models[data_type] = model
            self.forecasts[data_type] = forecast
            self.residuals[data_type] = merged
            
            report.update({
                'mae': merged['residual_abs'].mean(),
                'rmse': np.sqrt((merged['residual'] ** 2).mean()),
                'success': True
            })
            return forecast, report
        except Exception as e:
            report['error'] = str(e)
            return pd.DataFrame(), report
    
    def get_anomalies_from_residuals(self, data_type: str, threshold_std: float = 3.0):
        if data_type not in self.residuals:
            return pd.DataFrame()
        residuals = self.residuals[data_type]
        mean_res = residuals['residual'].mean()
        std_res = residuals['residual'].std()
        threshold = threshold_std * std_res
        anomalies = residuals[(residuals['residual'] > mean_res + threshold) |
                              (residuals['residual'] < mean_res - threshold)].copy()
        anomalies['anomaly_score'] = np.abs(anomalies['residual'] - mean_res) / std_res
        return anomalies

class BehaviorClusterer:
    """Cluster behavioral patterns using KMeans or DBSCAN"""
    
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.labels = {}
        self.reduced_features = {}
    
    def cluster_features(self, feature_matrix: pd.DataFrame, data_type: str,
                        method: str = 'kmeans', n_clusters: int = 3):
        report = {'data_type': data_type, 'method': method, 'success': False}
        try:
            if feature_matrix.empty:
                return np.array([]), report
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_matrix)
            self.scalers[data_type] = scaler
            
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            else:
                model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(features_scaled)
            self.models[data_type] = model
            self.labels[data_type] = labels
            
            if len(np.unique(labels)) > 1:
                report['silhouette_score'] = silhouette_score(features_scaled, labels)
                report['davies_bouldin'] = davies_bouldin_score(features_scaled, labels)
            
            unique, counts = np.unique(labels, return_counts=True)
            report.update({
                'n_clusters': len(unique),
                'cluster_sizes': dict(zip(unique.astype(int), counts.astype(int))),
                'success': True
            })
            return labels, report
        except Exception as e:
            report['error'] = str(e)
            return np.array([]), report
    
    def reduce_dimensions(self, feature_matrix: pd.DataFrame, data_type: str, method: str = 'pca'):
        if data_type not in self.scalers or feature_matrix.empty:
            return None
        features_scaled = self.scalers[data_type].transform(feature_matrix)
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        self.reduced_features[data_type] = reducer.fit_transform(features_scaled)
        return self.reduced_features[data_type]

class FeatureModelingPipeline:
    """Complete Milestone 2 pipeline"""
    
    def __init__(self):
        self.feature_extractor = TSFreshFeatureExtractor()
        self.trend_modeler = ProphetTrendModeler()
        self.clusterer = BehaviorClusterer()
    
    def run_pipeline(self, processed_data, window_size=60, forecast_periods=100,
                     clustering_method='kmeans', n_clusters=3):
        results = {
            'feature_matrices': {},
            'forecasts': {},
            'cluster_labels': {},
            'reports': {}
        }
        for data_type, df in processed_data.items():
            st.markdown(f"### 📊 {data_type.replace('_', ' ').title()}")
            
            with st.expander("🔵 Feature Extraction", expanded=True):
                feature_matrix, ext_report = self.feature_extractor.extract_features(df, data_type, window_size)
                if not feature_matrix.empty:
                    results['feature_matrices'][data_type] = feature_matrix
                    results['reports'][f'{data_type}_extraction'] = ext_report
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Features", ext_report['features_extracted'])
                    col2.metric("Windows", ext_report['feature_windows'])
                    col3.metric("Status", "✅ Success" if ext_report['success'] else "❌ Failed")
                    top_features = self.feature_extractor.get_top_features(5)
                    if not top_features.empty:
                        st.dataframe(top_features, width='stretch')
            
            with st.expander("🟡 Trend Modeling (Prophet)", expanded=True):
                forecast, mod_report = self.trend_modeler.fit_and_predict(df, data_type, forecast_periods)
                if not forecast.empty:
                    results['forecasts'][data_type] = forecast
                    results['reports'][f'{data_type}_modeling'] = mod_report
                    col1, col2 = st.columns(2)
                    col1.metric("MAE", f"{mod_report.get('mae',0):.2f}")
                    col2.metric("RMSE", f"{mod_report.get('rmse',0):.2f}")
                    self._plot_forecast(df, forecast, data_type)
                    anomalies = self.trend_modeler.get_anomalies_from_residuals(data_type)
                    if not anomalies.empty:
                        st.info(f"🚨 {len(anomalies)} anomalies detected")
                        st.dataframe(anomalies.head(5), width='stretch')
            
            with st.expander("🟢 Behavioral Clustering", expanded=True):
                if data_type in results['feature_matrices']:
                    feature_matrix = results['feature_matrices'][data_type]
                    labels, clust_report = self.clusterer.cluster_features(feature_matrix, data_type,
                                                                           clustering_method, n_clusters)
                    if len(labels) > 0:
                        results['cluster_labels'][data_type] = labels
                        results['reports'][f'{data_type}_clustering'] = clust_report
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Clusters", clust_report['n_clusters'])
                        col2.metric("Silhouette", f"{clust_report.get('silhouette_score',0):.3f}")
                        col3.metric("Davies-Bouldin", f"{clust_report.get('davies_bouldin',0):.3f}")
                        reduced = self.clusterer.reduce_dimensions(feature_matrix, data_type, 'pca')
                        if reduced is not None:
                            self._plot_clusters(reduced, labels, data_type)
            st.divider()
        return results
    
    def _plot_forecast(self, df_orig, forecast, data_type):
        metric_col = data_type
        if metric_col not in df_orig.columns:
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_orig['timestamp'], y=df_orig[metric_col],
                                 mode='markers', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                 mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                                 mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                                 mode='lines', fill='tonexty',
                                 fillcolor='rgba(236, 72, 153, 0.1)',
                                 line=dict(width=0), showlegend=False))
        fig.update_layout(title=f"Prophet Forecast - {metric_col.replace('_',' ').title()}",
                          xaxis_title="Time", yaxis_title=metric_col.replace('_',' ').title(),
                          height=400, template='plotly_white')
        st.plotly_chart(fig, width='stretch')
    
    def _plot_clusters(self, reduced_features, labels, data_type):
        fig = px.scatter(x=reduced_features[:,0], y=reduced_features[:,1], color=labels.astype(str),
                         title=f"Behavioral Clustering - {data_type.replace('_',' ').title()}",
                         color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, width='stretch')


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import datetime, timedelta
# from typing import Dict, List, Tuple, Optional
# import warnings
# warnings.filterwarnings('ignore')

# from tsfresh import extract_features
# from tsfresh.utilities.dataframe_functions import impute
# from tsfresh.feature_extraction import MinimalFCParameters
# from prophet import Prophet
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.metrics import silhouette_score, davies_bouldin_score

# # Custom Color Palette
# COLORS = {
#     'primary': '#6366F1',      # Indigo
#     'secondary': '#EC4899',    # Pink
#     'accent': '#10B981',       # Emerald
#     'warning': '#F59E0B',      # Amber
#     'danger': '#EF4444',       # Red
#     'dark': '#1F2937',         # Gray-800
#     'light': '#F3F4F6'         # Gray-100
# }

# class TSFreshFeatureExtractor:
#     """Extract time-series features using TSFresh"""
    
#     def __init__(self):
#         self.feature_matrix = None
#         self.feature_names = []
        
#     def extract_features(self, df: pd.DataFrame, data_type: str, 
#                         window_size: int = 60) -> Tuple[pd.DataFrame, Dict]:
#         """Extract statistical features from time-series data"""
        
#         report = {'data_type': data_type, 'success': False}
        
#         try:
#             # Prepare data
#             df_prepared = self._prepare_data(df, data_type, window_size)
#             if df_prepared is None:
#                 return pd.DataFrame(), report
            
#             # Extract features
#             feature_matrix = extract_features(
#                 df_prepared,
#                 column_id='window_id',
#                 column_sort='timestamp',
#                 default_fc_parameters=MinimalFCParameters(),
#                 disable_progressbar=True,
#                 n_jobs=1
#             )
            
#             feature_matrix = impute(feature_matrix)
#             feature_matrix = self._remove_constant_features(feature_matrix)
            
#             self.feature_matrix = feature_matrix
#             self.feature_names = list(feature_matrix.columns)
            
#             report.update({
#                 'features_extracted': len(self.feature_names),
#                 'feature_windows': len(feature_matrix),
#                 'success': True
#             })
            
#             return feature_matrix, report
            
#         except Exception as e:
#             report['error'] = str(e)
#             return pd.DataFrame(), report
    
#     def _prepare_data(self, df: pd.DataFrame, data_type: str, window_size: int):
#         """Prepare data in TSFresh format"""
#         metric_map = {
#             'heart_rate': 'heart_rate',
#             'step_count': 'step_count',
#             'duration_minutes': 'duration_minutes'
#         }

        
#         metric_col = metric_map.get(data_type)
#         if not metric_col or metric_col not in df.columns:
#             return None
        
#         df_sorted = df.sort_values('timestamp').reset_index(drop=True)
#         prepared_data = []
#         step_size = window_size // 2
        
#         for i in range(0, len(df_sorted) - window_size + 1, step_size):
#             window_data = df_sorted.iloc[i:i+window_size].copy()
#             window_data['window_id'] = i // step_size
#             prepared_data.append(window_data[['window_id', 'timestamp', metric_col]])
        
#         if not prepared_data:
#             return None
        
#         df_prepared = pd.concat(prepared_data, ignore_index=True)
#         df_prepared = df_prepared.rename(columns={metric_col: 'value'})
#         return df_prepared
    
#     def _remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Remove zero-variance features"""
#         constant_features = [col for col in df.columns if df[col].std() == 0]
#         return df.drop(columns=constant_features) if constant_features else df
    
#     def get_top_features(self, n: int = 10) -> pd.DataFrame:
#         """Get top N most variable features"""
#         if self.feature_matrix is None or self.feature_matrix.empty:
#             return pd.DataFrame()
        
#         feature_variance = self.feature_matrix.var().sort_values(ascending=False).head(n)
#         return pd.DataFrame({
#             'Feature': feature_variance.index,
#             'Variance': feature_variance.values,
#             'Mean': [self.feature_matrix[f].mean() for f in feature_variance.index]
#         })


# class ProphetTrendModeler:
#     """Model time-series trends using Facebook Prophet"""
    
#     def __init__(self):
#         self.models = {}
#         self.forecasts = {}
#         self.residuals = {}
    
#     def fit_and_predict(self, df: pd.DataFrame, data_type: str, 
#                        forecast_periods: int = 100) -> Tuple[pd.DataFrame, Dict]:
#         """Fit Prophet model and generate predictions"""
        
#         report = {'data_type': data_type, 'success': False}
        
#         try:
#             metric_map = {
#                 'heart_rate': 'heart_rate',
#                 'step_count': 'step_count',
#                 'duration_minutes': 'duration_minutes'
#             }

            
#             # metric_col = metric_map.get(data_type)
#             # if not metric_col or metric_col not in df.columns:
#             #     return pd.DataFrame(), report
#             # Replace above code with:
#             metric_col = data_type  # use the column name passed from app.py
#             if metric_col not in df.columns:
#                 report['error'] = f"Column '{metric_col}' not found in dataframe"
#                 return pd.DataFrame(), report

#             # Ensure timestamp is datetime and tz-naive
#             df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

#             # Prepare Prophet data
#             prophet_df = pd.DataFrame({
#                 'ds': df['timestamp'],
#                 'y': df[metric_col]
#             }).dropna()
            
#             if len(prophet_df) < 2:
#                 return pd.DataFrame(), report
            
#             # Fit model
#             model = Prophet(
#                 daily_seasonality=True,
#                 weekly_seasonality=False,
#                 yearly_seasonality=False,
#                 changepoint_prior_scale=0.05,
#                 interval_width=0.95
#             )
#             model.fit(prophet_df)
            
#             # Generate forecast
#             future = model.make_future_dataframe(periods=forecast_periods, freq='min')
#             forecast = model.predict(future)
            
#             # Calculate residuals
#             merged = prophet_df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
#                                      on='ds', how='left')
#             merged['residual'] = merged['y'] - merged['yhat']
#             merged['residual_abs'] = np.abs(merged['residual'])
            
#             self.models[data_type] = model
#             self.forecasts[data_type] = forecast
#             self.residuals[data_type] = merged
            
#             report.update({
#                 'mae': merged['residual_abs'].mean(),
#                 'rmse': np.sqrt((merged['residual'] ** 2).mean()),
#                 'success': True
#             })
            
#             return forecast, report
            
#         except Exception as e:
#             report['error'] = str(e)
#             return pd.DataFrame(), report
    
#     def get_anomalies_from_residuals(self, data_type: str, threshold_std: float = 3.0):
#         """Identify anomalies based on residuals"""
#         if data_type not in self.residuals:
#             return pd.DataFrame()
        
#         residuals = self.residuals[data_type]
#         mean_res = residuals['residual'].mean()
#         std_res = residuals['residual'].std()
#         threshold = threshold_std * std_res
        
#         anomalies = residuals[
#             (residuals['residual'] > mean_res + threshold) |
#             (residuals['residual'] < mean_res - threshold)
#         ].copy()
        
#         anomalies['anomaly_score'] = np.abs(anomalies['residual'] - mean_res) / std_res
#         return anomalies


# class BehaviorClusterer:
#     """Cluster behavioral patterns using KMeans and DBSCAN"""
    
#     def __init__(self):
#         self.scalers = {}
#         self.models = {}
#         self.labels = {}
#         self.reduced_features = {}
    
#     def cluster_features(self, feature_matrix: pd.DataFrame, data_type: str,
#                         method: str = 'kmeans', n_clusters: int = 3) -> Tuple[np.ndarray, Dict]:
#         """Cluster feature vectors"""
        
#         report = {'data_type': data_type, 'method': method, 'success': False}
        
#         try:
#             if feature_matrix.empty:
#                 return np.array([]), report
            
#             # Standardize
#             scaler = StandardScaler()
#             features_scaled = scaler.fit_transform(feature_matrix)
#             self.scalers[data_type] = scaler
            
#             # Cluster
#             if method == 'kmeans':
#                 model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#             else:  # dbscan
#                 model = DBSCAN(eps=0.5, min_samples=5)
            
#             labels = model.fit_predict(features_scaled)
#             self.models[data_type] = model
#             self.labels[data_type] = labels
            
#             # Metrics
#             if len(np.unique(labels)) > 1:
#                 report['silhouette_score'] = silhouette_score(features_scaled, labels)
#                 report['davies_bouldin'] = davies_bouldin_score(features_scaled, labels)
            
#             unique, counts = np.unique(labels, return_counts=True)
#             report.update({
#                 'n_clusters': len(unique),
#                 'cluster_sizes': dict(zip(unique.astype(int), counts.astype(int))),
#                 'success': True
#             })
            
#             return labels, report
            
#         except Exception as e:
#             report['error'] = str(e)
#             return np.array([]), report
    
#     def reduce_dimensions(self, feature_matrix: pd.DataFrame, data_type: str, method: str = 'pca'):
#         """Reduce dimensions for visualization"""
#         if data_type not in self.scalers or feature_matrix.empty:
#             return None
        
#         features_scaled = self.scalers[data_type].transform(feature_matrix)
        
#         if method == 'pca':
#             reducer = PCA(n_components=2, random_state=42)
#         else:  # tsne
#             reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        
#         self.reduced_features[data_type] = reducer.fit_transform(features_scaled)
#         return self.reduced_features[data_type]


# class FeatureModelingPipeline:
#     """Complete Milestone 2 pipeline"""
    
#     def __init__(self):
#         self.feature_extractor = TSFreshFeatureExtractor()
#         self.trend_modeler = ProphetTrendModeler()
#         self.clusterer = BehaviorClusterer()
    
#     def run_pipeline(self, processed_data: Dict[str, pd.DataFrame], 
#                     window_size: int = 60, forecast_periods: int = 100,
#                     clustering_method: str = 'kmeans', n_clusters: int = 3) -> Dict:
#         """Run complete Milestone 2 pipeline"""
        
#         results = {
#             'feature_matrices': {},
#             'forecasts': {},
#             'cluster_labels': {},
#             'reports': {}
#         }
        
#         for data_type, df in processed_data.items():
#             st.markdown(f"### 📊 {data_type.replace('_', ' ').title()}")
            
#             # Feature Extraction
#             with st.expander("🔵 Feature Extraction", expanded=True):
#                 feature_matrix, ext_report = self.feature_extractor.extract_features(
#                     df, data_type, window_size
#                 )
                
#                 if not feature_matrix.empty:
#                     results['feature_matrices'][data_type] = feature_matrix
#                     results['reports'][f'{data_type}_extraction'] = ext_report
                    
#                     col1, col2, col3 = st.columns(3)
#                     col1.metric("Features", ext_report['features_extracted'])
#                     col2.metric("Windows", ext_report['feature_windows'])
#                     col3.metric("Status", "✅ Success" if ext_report['success'] else "❌ Failed")
                    
#                     # Top features
#                     top_features = self.feature_extractor.get_top_features(5)
#                     if not top_features.empty:
#                         st.dataframe(top_features, width='stretch')
            
#             # Prophet Modeling
#             with st.expander("🟡 Trend Modeling (Prophet)", expanded=True):
#                 forecast, mod_report = self.trend_modeler.fit_and_predict(
#                     df, data_type, forecast_periods
#                 )
                
#                 if not forecast.empty:
#                     results['forecasts'][data_type] = forecast
#                     results['reports'][f'{data_type}_modeling'] = mod_report
                    
#                     col1, col2 = st.columns(2)
#                     col1.metric("MAE", f"{mod_report.get('mae', 0):.2f}")
#                     col2.metric("RMSE", f"{mod_report.get('rmse', 0):.2f}")
                    
#                     # Forecast visualization
#                     self._plot_forecast(df, forecast, data_type)
                    
#                     # Anomalies
#                     anomalies = self.trend_modeler.get_anomalies_from_residuals(data_type)
#                     if not anomalies.empty:
#                         st.info(f"🚨 {len(anomalies)} anomalies detected")
#                         st.dataframe(anomalies.head(5), width='stretch')
            
#             # Clustering
#             with st.expander("🟢 Behavioral Clustering", expanded=True):
#                 if data_type in results['feature_matrices']:
#                     feature_matrix = results['feature_matrices'][data_type]
                    
#                     labels, clust_report = self.clusterer.cluster_features(
#                         feature_matrix, data_type, clustering_method, n_clusters
#                     )
                    
#                     if len(labels) > 0:
#                         results['cluster_labels'][data_type] = labels
#                         results['reports'][f'{data_type}_clustering'] = clust_report
                        
#                         col1, col2, col3 = st.columns(3)
#                         col1.metric("Clusters", clust_report['n_clusters'])
#                         col2.metric("Silhouette", f"{clust_report.get('silhouette_score', 0):.3f}")
#                         col3.metric("Davies-Bouldin", f"{clust_report.get('davies_bouldin', 0):.3f}")
                        
#                         # Cluster visualization
#                         reduced = self.clusterer.reduce_dimensions(feature_matrix, data_type, 'pca')
#                         if reduced is not None:
#                             self._plot_clusters(reduced, labels, data_type)
            
#             st.divider()
        
#         self._show_summary(results)
#         return results
    
#     def _plot_forecast(self, df_orig, forecast, data_type):
#         """Plot Prophet forecast"""
#         metric_col = data_type  # use the actual column name
#         if metric_col not in df_orig.columns:
#             return

        
#         if not metric_col or metric_col not in df_orig.columns:
#             return
        
#         fig = go.Figure()
        
#         # Actual
#         fig.add_trace(go.Scatter(
#             x=df_orig['timestamp'], y=df_orig[metric_col],
#             mode='markers', name='Actual',
#             marker=dict(size=4, color=COLORS['primary'])
#         ))
        
#         # Forecast
#         fig.add_trace(go.Scatter(
#             x=forecast['ds'], y=forecast['yhat'],
#             mode='lines', name='Forecast',
#             line=dict(color=COLORS['secondary'], width=2)
#         ))
        
#         # Confidence interval
#         fig.add_trace(go.Scatter(
#             x=forecast['ds'], y=forecast['yhat_upper'],
#             mode='lines', line=dict(width=0), showlegend=False
#         ))
#         fig.add_trace(go.Scatter(
#             x=forecast['ds'], y=forecast['yhat_lower'],
#             mode='lines', fill='tonexty',
#             fillcolor=f'rgba(236, 72, 153, 0.1)',
#             line=dict(width=0), showlegend=False
#         ))
        
#         fig.update_layout(
#             title=f"Prophet Forecast - {data_type.title()}",
#             xaxis_title="Time", yaxis_title=metric_col.replace('_', ' ').title(),
#             height=400, template='plotly_white'
#         )
        
#         st.plotly_chart(fig, width='stretch')
    
#     def _plot_clusters(self, reduced, labels, data_type):
#         """Plot cluster visualization"""
#         df_viz = pd.DataFrame({
#             'PC1': reduced[:, 0], 'PC2': reduced[:, 1],
#             'Cluster': labels.astype(str)
#         })
        
#         fig = px.scatter(
#             df_viz, x='PC1', y='PC2', color='Cluster',
#             title=f"Cluster Visualization - {data_type.title()}",
#             color_discrete_sequence=[COLORS['primary'], COLORS['accent'], 
#                                    COLORS['warning'], COLORS['secondary']]
#         )
#         fig.update_traces(marker=dict(size=8, opacity=0.7))
#         fig.update_layout(height=400, template='plotly_white')
        
#         st.plotly_chart(fig, width='stretch')
    
#     def _show_summary(self, results):
#         """Show summary report"""
#         st.markdown("---")
#         st.header("📝 Milestone 2 Summary")
        
#         col1, col2, col3, col4 = st.columns(4)
        
#         col1.metric("Data Types", len(results['feature_matrices']), 
#                    delta_color="off")
#         col2.metric("Total Features", 
#                    sum(len(fm.columns) for fm in results['feature_matrices'].values()),
#                    delta_color="off")
#         col3.metric("Models Trained", len(results['forecasts']),
#                    delta_color="off")
#         col4.metric("Clusterings", len(results['cluster_labels']),
#                    delta_color="off")
        
#         st.success("""
#         ✅ **Milestone 2 Complete**
#         - TSFresh feature extraction
#         - Prophet trend modeling
#         - Behavioral clustering (KMeans/DBSCAN)
#         - Anomaly detection from residuals
#         """)


# def main():
#     st.set_page_config(page_title="FitPulse Milestone 2", page_icon="🔬", layout="wide")
    
#     # Custom CSS
#     st.markdown(f"""
#     <style>
#         .stMetric {{
#             background: linear-gradient(135deg, {COLORS['primary']}15 0%, {COLORS['accent']}15 100%);
#             padding: 1rem;
#             border-radius: 0.5rem;
#         }}
#         .stExpander {{
#             border-left: 3px solid {COLORS['primary']};
#         }}
#     </style>
#     """, unsafe_allow_html=True)
    
#     st.title("🔬 FitPulse Milestone 2")
#     st.markdown("**Feature Extraction • Trend Modeling • Behavioral Clustering**")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         window_size = st.slider("Feature Window (min)", 10, 120, 60, 10)
#         forecast_periods = st.slider("Forecast Periods", 50, 500, 100, 50)
#         clustering_method = st.selectbox("Clustering", ['kmeans', 'dbscan'])
#         n_clusters = st.slider("Number of Clusters", 2, 10, 3) if clustering_method == 'kmeans' else 3
#         use_sample = st.checkbox("Use Sample Data", True)
        
#         st.divider()
#         st.info("💡 Adjust parameters to customize the analysis pipeline")
    
#     # Initialize
#     if 'pipeline' not in st.session_state:
#         st.session_state.pipeline = FeatureModelingPipeline()
    
#     # Run button
#     if st.button("🚀 Run Milestone 2 Pipeline", type="primary", width='stretch'):
#         with st.spinner("Processing..."):
#             uploaded_files = st.file_uploader("Upload CSV/JSON files", accept_multiple_files=True)
#             processed_data = load_and_process(uploaded_files)  # your existing preprocessing function

            
#             results = st.session_state.pipeline.run_pipeline(
#                 processed_data=data,
#                 window_size=window_size,
#                 forecast_periods=forecast_periods,
#                 clustering_method=clustering_method,
#                 n_clusters=n_clusters
#             )
            
#             st.session_state.results = results
#             st.balloons()


# if __name__ == "__main__":
#     main()