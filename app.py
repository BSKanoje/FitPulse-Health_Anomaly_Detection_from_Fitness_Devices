import streamlit as st
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from pipeline.preprocessor import FitnessDataPreprocessor
import logging
import warnings

# -----------------------------
# Setup
# -----------------------------
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

st.set_page_config(page_title="FitPulse", layout="wide")
st.title("FitPulse - Health Anomaly Detection")

# -----------------------------
# Initialize Session State
# -----------------------------
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = FitnessDataPreprocessor()
    st.session_state.processed = None
    st.session_state.features = None
    st.session_state.forecast = None
    st.session_state.anomalies = None
    st.session_state.cluster_labels = None

# -----------------------------
# Module 1: Preprocessing
# -----------------------------
st.header("📁 Step 1: Preprocessing")

uploaded_files = st.file_uploader(
    "Upload CSV/JSON files", type=['csv', 'json'], accept_multiple_files=True
)

target_freq = st.selectbox("Target Frequency", ['1min','5min','15min','30min','1hour'])
fill_method = st.selectbox(
    "Missing Value Fill Method",
    ['interpolate','forward_fill','backward_fill','zero','drop']
)

if st.button("🚀 Run Preprocessing Pipeline"):
    if uploaded_files:
        try:
            with st.spinner("Running preprocessing..."):
                st.session_state.processed = st.session_state.preprocessor.run_complete_pipeline(
                    uploaded_files=uploaded_files,
                    target_frequency=target_freq,
                    fill_method=fill_method
                )
            st.success("✅ Preprocessing complete!")
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
    else:
        st.error("❌ Please upload at least one CSV or JSON file.")

st.markdown("---")

# -----------------------------
# Milestone 2 Classes
# -----------------------------
# Copy all Milestone 2 code here: TSFreshFeatureExtractor, ProphetTrendModeler,
# BehaviorClusterer, FeatureModelingPipeline
# (The code you pasted above goes here)

from milestone2 import FeatureModelingPipeline  # If saved separately


# -----------------------------
# Module 2: Full Analysis
# -----------------------------
if st.session_state.processed:
    st.header("📊 Step 2: Full Analysis (Feature Extraction + Modeling + Clustering)")

    # Combine all processed data
    try:
        combined_df = pd.concat(st.session_state.processed.values(), ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp']).dt.tz_localize(None)
    except Exception as e:
        st.error(f"Error combining processed data: {e}")
        st.stop()

    numeric_columns = list(set(combined_df.select_dtypes(include=['number']).columns))
    numeric_columns = [c for c in numeric_columns if not c.endswith('_outlier')]

    if numeric_columns:
        selected_metric = st.selectbox("Select metric to analyze", numeric_columns)

        if st.button("⚡ Run Full Analysis"):
            with st.spinner("Running full Milestone 2 analysis..."):
                # Initialize Milestone 2 pipeline if not already
                if 'pipeline_m2' not in st.session_state:
                    st.session_state.pipeline_m2 = FeatureModelingPipeline()

                # Use actual processed data
                data_dict = {selected_metric: combined_df[['timestamp', selected_metric]]}

                results = st.session_state.pipeline_m2.run_pipeline(
                    processed_data=data_dict,
                    window_size=60,
                    forecast_periods=100,
                    clustering_method='kmeans',
                    n_clusters=3
                )
                st.session_state.results = results
                st.success("✅ Full Analysis Complete!")
                st.balloons()
    else:
        st.info("No numeric columns available for analysis.")
else:
    st.info("Run preprocessing first to enable full analysis.")




# early 
# import streamlit as st
# import pandas as pd
# import numpy as np
# from tsfresh import extract_features
# from tsfresh.feature_extraction import ComprehensiveFCParameters
# from tsfresh.utilities.dataframe_functions import impute
# from prophet import Prophet
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import plotly.express as px
# import plotly.graph_objects as go
# from pipeline.preprocessor import FitnessDataPreprocessor
# import logging
# import warnings

# # -----------------------------
# # Setup
# # -----------------------------
# logging.getLogger("streamlit").setLevel(logging.ERROR)
# warnings.filterwarnings("ignore")

# st.set_page_config(page_title="FitPulse", layout="wide")
# st.title("FitPulse - Health Anomaly Detection")

# # -----------------------------
# # Initialize Session State
# # -----------------------------
# if 'preprocessor' not in st.session_state:
#     st.session_state.preprocessor = FitnessDataPreprocessor()
#     st.session_state.processed = None
#     st.session_state.features = None
#     st.session_state.forecast = None
#     st.session_state.anomalies = None
#     st.session_state.cluster_labels = None

# # -----------------------------
# # Module 1: Preprocessing
# # -----------------------------
# st.header("📁 Step 1: Preprocessing")

# uploaded_files = st.file_uploader(
#     "Upload CSV/JSON files", type=['csv', 'json'], accept_multiple_files=True
# )

# target_freq = st.selectbox("Target Frequency", ['1min','5min','15min','30min','1hour'])
# fill_method = st.selectbox(
#     "Missing Value Fill Method",
#     ['interpolate','forward_fill','backward_fill','zero','drop']
# )

# if st.button("🚀 Run Preprocessing Pipeline"):
#     if uploaded_files:
#         try:
#             with st.spinner("Running preprocessing..."):
#                 st.session_state.processed = st.session_state.preprocessor.run_complete_pipeline(
#                     uploaded_files=uploaded_files,
#                     target_frequency=target_freq,
#                     fill_method=fill_method
#                 )
#             st.success("✅ Preprocessing complete!")
#         except Exception as e:
#             st.error(f"Error during preprocessing: {e}")
#     else:
#         st.error("❌ Please upload at least one CSV or JSON file.")

# st.markdown("---")

# # -----------------------------
# # Module 2: Full Analysis
# # -----------------------------
# if st.session_state.processed:
#     st.header("📊 Step 2: Full Analysis (Feature Extraction + Modeling + Clustering)")

#     # Combine all processed data
#     try:
#         combined_df = pd.concat(st.session_state.processed.values(), ignore_index=True)
#         combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
#     except Exception as e:
#         st.error(f"Error combining processed data: {e}")
#         st.stop()

#     numeric_columns = list(set(combined_df.select_dtypes(include=['number']).columns))
#     numeric_columns = [c for c in numeric_columns if not c.endswith('_outlier')]

#     if numeric_columns:
#         selected_metric = st.selectbox("Select metric to analyze", numeric_columns)

#         if st.button("⚡ Run Full Analysis"):
#             try:
#                 # -----------------------------
#                 # Feature Extraction
#                 # -----------------------------
#                 st.subheader("1️⃣ Feature Extraction")
#                 df_for_features = combined_df[['timestamp', 'source_file'] + numeric_columns].copy()
#                 df_for_features['timestamp'] = pd.to_datetime(df_for_features['timestamp'])

#                 features = extract_features(
#                     df_for_features,
#                     column_id='source_file',
#                     column_sort='timestamp',
#                     default_fc_parameters=ComprehensiveFCParameters()
#                 )
#                 features = impute(features)
#                 st.session_state.features = features
#                 st.dataframe(features.head(10))

#                 # -----------------------------
#                 # Prophet Modeling
#                 # -----------------------------
#                 st.subheader("2️⃣ Trend & Seasonality Modeling (Prophet)")
#                 df_metric = combined_df[['timestamp', selected_metric]].rename(
#                     columns={'timestamp':'ds', selected_metric:'y'}
#                 ).dropna()

#                 # Remove timezone
#                 df_metric['ds'] = pd.to_datetime(df_metric['ds']).dt.tz_localize(None)

#                 # Aggregate duplicates
#                 df_metric = df_metric.groupby('ds').mean().reset_index()

#                 if len(df_metric) < 2:
#                     st.warning("Not enough data points for Prophet modeling.")
#                 else:
#                     model = Prophet(daily_seasonality=True)
#                     model.fit(df_metric)
#                     forecast = model.predict(df_metric)
#                     forecast['residual'] = df_metric['y'].values - forecast['yhat'].values
#                     st.session_state.forecast = forecast

#                     fig = go.Figure()
#                     fig.add_trace(go.Scatter(x=df_metric['ds'], y=df_metric['y'], mode='lines+markers', name='Actual'))
#                     fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
#                     st.plotly_chart(fig, width='stretch')

#                 # -----------------------------
#                 # Cluster Visualization (PCA + KMeans)
#                 # -----------------------------
#                 st.subheader("3️⃣ Cluster Visualization (PCA + KMeans)")
#                 features_filled = features.fillna(0)
#                 X_scaled = StandardScaler().fit_transform(features_filled)

#                 n_samples = X_scaled.shape[0]
#                 n_clusters = min(3, n_samples) if n_samples > 1 else 1

#                 if n_samples < 2:
#                     st.warning("Not enough samples for clustering. Skipping cluster visualization.")
#                 else:
#                     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#                     labels = kmeans.fit_predict(X_scaled)
#                     st.session_state.cluster_labels = labels

#                     pca = PCA(n_components=2)
#                     components = pca.fit_transform(X_scaled)
#                     features_filled['PC1'] = components[:,0]
#                     features_filled['PC2'] = components[:,1]
#                     features_filled['cluster'] = labels

#                     fig = px.scatter(
#                         features_filled, x='PC1', y='PC2', color='cluster',
#                         title=f"Feature Clustering with PCA (n_clusters={n_clusters})",
#                         hover_data=[features_filled.columns[0]]
#                     )
#                     st.plotly_chart(fig, width='stretch')

#             except Exception as e:
#                 st.error(f"Error during full analysis: {e}")

#     else:
#         st.info("No numeric columns available for analysis.")
# else:
#     st.info("Run preprocessing first to enable full analysis.")
