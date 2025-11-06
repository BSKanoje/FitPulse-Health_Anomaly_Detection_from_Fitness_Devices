import streamlit as st
import pandas as pd
from src.utils.data_preprocessing import preprocess_data
from src.utils.feature_extraction import extract_all_features, model_with_prophet
from src.models.anomaly_detection import detect_anomalies
from src.utils.visualization import plot_anomalies, plot_trends
from src.utils.utils import export_report
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="FitPulse", layout="wide")

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown(
    """
    <style>
    /* Target all tab labels */
    .stTabs [data-baseweb="tab"] {
        color: black !important;
        font-weight: 600 !important;
    }

    /* Active tab highlight (underline) */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: black !important;
    }

    /* Optional: make active tab text darker or bold */
    .stTabs [aria-selected="true"] {
        color: black !important;
        font-weight: 700 !important;
    }
    .stApp {
        background-color: #E6E6FA;
        color: #4B0082;
    }
    .stTitle {
        color: #4B0082 !important;
        font-size: 40px !important;
        font-weight: bold !important;
    }
    [data-testid="stSidebar"] {
        background-color: #4B0082 !important;
        color: #D8BFD8;
    }
    div.stButton > button:first-child {
        background-color: #9370DB;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 40px;
    }
    .stMetricLabel, .stMetricValue {
        color: #4B0082 !important;
        font-weight: bold;
    }
    
    /* Make all tab labels black */
    div[data-baseweb="tab"] > button {
        color: black !important;
        font-weight: 600 !important;
    }
    /* Optional: change active tab underline color */
    div[data-baseweb="tab-highlight"] {
        background-color: black !important;
    }

    /* Style ONLY the Download PDF Report button */
    div.stDownloadButton > button {
        background-color: #28a745 !important;  
        color: white !important;              
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 8px 15px !important;
        border: 1px solid #1e7e34 !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #218838 !important;   
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Custom Alert Boxes
# ----------------------------

def info_box(message, bg="#FFFACD"):
    st.markdown(
        f"""
        <div style="background-color:{bg};color:black;padding:12px 20px;
        border-radius:8px;border:1px solid #ccc;font-weight:bold;margin-bottom:10px;">
        {message.replace('\n', '<br>')}
        </div>
        """,
        unsafe_allow_html=True
    )

def success_box(message):
    st.markdown(
        f"""
        <div style="background-color:#d4edda;color:black;padding:12px 20px;
        border-radius:8px;border:1px solid #c3e6cb;font-weight:bold;margin-bottom:10px;">
        {message}
        </div>
        """,
        unsafe_allow_html=True
    )

def red_alert(message):
    st.markdown(
        f"""
        <div style="background-color:#f8d7da;color:black;padding:12px 20px;
        border-radius:8px;border:1px solid #f5c6cb;font-weight:bold;margin-bottom:10px;">
        {message.replace('\n', '<br>')}
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Page Title
# ----------------------------
st.title("FitPulse Health Anomaly Detection from Fitness Devices")
st.markdown(
    "Upload one or more fitness data files (CSV/JSON) to detect anomalies in heart rate, steps, and sleep."
)

# ----------------------------
# Sidebar Upload
# ----------------------------
st.sidebar.image("assets/watch.png", width='stretch')
st.sidebar.header("Uploads")

uploaded_files = st.sidebar.file_uploader(
    "Upload your CSV/JSON files",
    type=['csv', 'json'],
    accept_multiple_files=True
)

# ----------------------------
# Data Loading & Preprocessing
# ----------------------------
@st.cache_data
def load_and_process(files):
    if not files:
        info_box("Please upload at least one file to proceed.")
        st.stop()

    dfs = []
    for uploaded_file in files:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)

            # ✅ Clean column names: remove spaces and make lowercase
            df.columns = df.columns.str.strip().str.lower()

        except Exception as e:
            red_alert(f"❌ Error reading {uploaded_file.name}: {e}")
            continue

        # ✅ Ensure timestamp column exists after cleaning
        if 'timestamp' not in df.columns:
            info_box(f"No 'timestamp' found in {uploaded_file.name}. Generating dummy timestamps.")
            df.insert(0, 'timestamp', pd.date_range(start='2025-01-01', periods=len(df), freq='H'))

        df['source_file'] = uploaded_file.name
        dfs.append(df)


    if not dfs:
        red_alert("No valid data to process.")
        st.stop()

    combined_df = pd.concat(dfs, ignore_index=True)
    processed = preprocess_data(combined_df)
    st.session_state.df = processed
    return processed

# ----------------------------
# Load Data
# ----------------------------
if 'df' not in st.session_state or uploaded_files:
    df = load_and_process(uploaded_files)
else:
    df = st.session_state.df

# ✅ Ensure 'timestamp' column exists or is created
if 'timestamp' not in df.columns:
    # Try to automatically detect a timestamp-like column
    possible_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    if possible_cols:
        df = df.rename(columns={possible_cols[0]: 'timestamp'})
        st.info(f"🕒 Renamed '{possible_cols[0]}' to 'timestamp'.")
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        #st.success("✅ Using existing 'timestamp' column from file.")
    else:
        st.error("❌ No timestamp or datetime column found in the uploaded file.")
        st.stop()

# ✅ Ensure timestamp is in datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp']).reset_index(drop=True)


# ----------------------------
# Data Display
# ----------------------------
if df is not None and len(df) > 0:
    success_box(f"✅ Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    st.dataframe(df.head())

    # ----------------------------
    # Feature Extraction & Prophet Models
    # ----------------------------
    with st.spinner("Extracting features and training models..."):
        features = extract_all_features(df)
        models = {}
        forecasts = {}
        for metric in ['heart_rate', 'steps', 'sleep_duration']:
            model, forecast, residuals = model_with_prophet(df, metric)
            models[metric] = (model, forecast, residuals)
            forecasts[metric] = forecast

    # ----------------------------
    # Anomaly Detection
    # ----------------------------
    with st.spinner("Detecting anomalies..."):
        anomalies, alerts = detect_anomalies(df, features, models)

    # ----------------------------
    # Display Results
    # ----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Anomaly Summary")
        if not anomalies.empty:
            st.dataframe(anomalies, width='stretch')
            st.markdown(
                f"<div style='font-size:26px; font-weight:bold; color:#4B0082;'>"
                f"Total Anomalies: {len(anomalies)}</div>", unsafe_allow_html=True
            )
        else:
            info_box("No anomalies detected. Your data looks great!")

    with col2:
        st.subheader("Alerts")

        if alerts:
            # If alerts is a dict (e.g., {'heart_rate': [...], 'steps': [...]})
            if isinstance(alerts, dict):
                for metric, alert_list in alerts.items():
                    for alert in alert_list:
                        red_alert(f"⚠️ {metric.capitalize()} — {alert}")

            # If alerts is a simple list
            elif isinstance(alerts, list):
                for alert in alerts:
                    red_alert(f"⚠️ {alert}")

            # Fallback (unexpected type)
            else:
                red_alert(f"⚠️ Unexpected alert format: {alerts}")

        else:
            success_box("All metrics normal — no anomalies detected!")

    # ----------------------------
    # 📄 Export & Download Report Section
    # ----------------------------
    st.subheader("Export FitPulse Report")

    if st.button("Generate Report"):
        try:
            # Generate both CSV and PDF inside the /report folder
            export_report(df, anomalies, alerts, filename="fitpulse_report", folder="report")

            pdf_path = "report/fitpulse_report.pdf"

            # Read PDF
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            st.markdown(
                """
                <div style="
                    background-color:#d4edda;      /* light green success box */
                    color:black;                    /* ✅ black text */
                    padding:12px 20px;
                    border-radius:8px;
                    border:1px solid #c3e6cb;       /* green border */
                    font-weight:bold;
                    margin-bottom:10px;
                ">
                    ✅ Report generated successfully!
                </div>
                """,
                unsafe_allow_html=True
            )

            # Show download button
            st.download_button(
                label="⬇ Download PDF Report",
                data=pdf_bytes,
                file_name="FitPulse_Health_Report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"❌ Failed to generate report: {e}")


    # =======================================
    # 📊 ADVANCED ANOMALY VISUALIZATION DASHBOARD
    # =======================================


    st.subheader("🔍 FitPulse Anomaly Insights Dashboard")

    # Tabs for visualization
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Anomaly Analysis (Timeline)",
        "📊 Rule-Based Thresholds",
        "🌀 Model-Based Clusters",
        "📈 Combined Summary & Trends"
    ])

    st.markdown("""
        <style>
        /* General tab text */
        [data-testid="stHorizontalBlock"] button p {
            color: black !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            transition: color 0.2s ease-in-out;
        }

        /* On hover */
        [data-testid="stHorizontalBlock"] button:hover p {
            color: black !important;
        }

        /* On click (active) */
        [data-testid="stHorizontalBlock"] button:active p {
            color: black !important;
        }

        /* When selected tab */
        [data-testid="stHorizontalBlock"] button[aria-selected="true"] p {
            color: black !important;
        }

        /* Optional: Change selected tab background */
        [data-testid="stHorizontalBlock"] button[aria-selected="true"] {
            background-color: #E6E6FA !important;  /* Lavender background */
            border-radius: 8px 8px 0 0 !important;
            border-bottom: 2px solid #4B0082 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Works whether anomalies is a list or DataFrame
    if anomalies is not None and len(anomalies) > 0:
        if isinstance(anomalies, list):
            anomalies_df = pd.DataFrame(anomalies)
        else:
            anomalies_df = anomalies.copy()
    else:
        st.warning("No anomalies detected to visualize.")
        anomalies_df = pd.DataFrame(columns=['timestamp', 'metric', 'value'])

    # -------------------------------------------------------
    # TAB 1: ANOMALY TIMELINE (THEMED FITPULSE STYLE)
    # -------------------------------------------------------
    with tab1:
        st.markdown("### 📅 Anomaly Detection Timeline")

        st.write("Visualizing individual metrics over time with anomaly highlights (FitPulse theme).")

        if 'timestamp' not in df.columns:
            st.warning("⚠️ No timestamp column available to plot timeline.")
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp')

            attributes = ['heart_rate', 'sleep_duration', 'steps']

            for col in attributes:
                if col not in df.columns:
                    st.warning(f"⚠️ Column '{col}' not found in data.")
                    continue

                # --- Plotly chart for better themed visuals ---
                fig = px.line(
                    df,
                    x='timestamp',
                    y=col,
                    title=f"{col.replace('_', ' ').capitalize()} Over Time",
                    markers=True,
                    line_shape='spline',
                    color_discrete_sequence=['#4B0082']  # Indigo FitPulse color
                )

                # Highlight anomalies for that metric (if exist)
                if not anomalies.empty and col in anomalies['metric'].values:
                    anomaly_points = anomalies[anomalies['metric'] == col]
                    fig.add_scatter(
                        x=anomaly_points['timestamp'],
                        y=anomaly_points['value'],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='#FF4500', size=10, symbol='circle'),
                    )

                fig.update_layout(
                    paper_bgcolor="#C49BFF",
                    plot_bgcolor="#C49BFF",
                    font=dict(color="#4B0082"),
                    xaxis_title="Timestamp",
                    yaxis_title=col.replace("_", " ").capitalize(),
                    title_font=dict(size=20, color="#4B0082", family="Arial"),
                    height=400
                )

                st.plotly_chart(fig, width='stretch')



    # -------------------------------------------------------
    # TAB 2: RULE-BASED ANOMALIES (THRESHOLDS)
    # -------------------------------------------------------
    with tab2:
        st.markdown("### 📊 Rule-Based Anomalies (Fixed Thresholds)")
        

        # Apply simple rule-based logic (independent of timestamp)
        rule_anoms = pd.DataFrame(columns=['metric', 'value'])
        if 'heart_rate' in df.columns:
            hr_anoms = df[(df['heart_rate'] > 110) | (df['heart_rate'] < 50)]
            if not hr_anoms.empty:
                temp = hr_anoms[['heart_rate']].rename(columns={'heart_rate': 'value'})
                temp['metric'] = 'heart_rate'
                rule_anoms = pd.concat([rule_anoms, temp])
        if 'steps' in df.columns:
            st_anoms = df[(df['steps'] > 2000) | (df['steps'] < 0)]
            if not st_anoms.empty:
                temp = st_anoms[['steps']].rename(columns={'steps': 'value'})
                temp['metric'] = 'steps'
                rule_anoms = pd.concat([rule_anoms, temp])
        if 'sleep_duration' in df.columns:
            sl_anoms = df[(df['sleep_duration'] < 60) | (df['sleep_duration'] > 600)]
            if not sl_anoms.empty:
                temp = sl_anoms[['sleep_duration']].rename(columns={'sleep_duration': 'value'})
                temp['metric'] = 'sleep_duration'
                rule_anoms = pd.concat([rule_anoms, temp])

        if not rule_anoms.empty:
            st.markdown(
                f"""
                <div style="
                    background-color:#E6E6FA;
                    color:#000000;
                    padding:15px;
                    border-radius:10px;
                    text-align:center;
                    border:1px solid #4B0082;
                    box-shadow:2px 2px 5px rgba(75, 0, 130, 0.2);
                    width:100%;
                ">
                    <h4 style="margin:0; color:#000000;">Total Rule-Based Anomalies</h4>
                    <h2 style="margin:0; color:#000000;">{len(rule_anoms)}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            fig_rule = px.scatter(
                rule_anoms,
                x='metric', y='value',
                color='metric',
                title="Detected Rule-Based Anomalies by Metric",
                color_discrete_sequence=['#FF4500', '#32CD32', '#1E90FF'],
                symbol='metric'
            )
            fig_rule.update_layout(
                paper_bgcolor="#C49BFF",
                plot_bgcolor="#C49BFF",
                font=dict(color="#4B0082"),
                height=500
            )
            st.plotly_chart(fig_rule, width='stretch')
        else:
            success_box("No anomalies detected by threshold rules 🎉")


    # -------------------------------------------------------
    # TAB 3: MODEL-BASED ANOMALIES (CLUSTERS)
    # -------------------------------------------------------
    with tab3:
        st.markdown("### 🌀 Model-Based Anomalies (Residual Clustering)")

        # Use metric values as features
        model_features = df[['heart_rate', 'steps', 'sleep_duration']].fillna(0)
        X = (model_features - model_features.mean()) / model_features.std()

        if len(X) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X)
            X['cluster'] = clusters
            X['timestamp'] = df['timestamp'] if 'timestamp' in df.columns else df.index

            # Plot clusters
            fig_clusters = px.scatter_3d(
                X,
                x='heart_rate', y='steps', z='sleep_duration',
                color='cluster',
                title="Model-Based Clustering of Fitness Data",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_clusters.update_layout(
                paper_bgcolor="#C49BFF",
                font=dict(color="#4B0082"),
                height=600
            )
            st.plotly_chart(fig_clusters, width='stretch')

            cluster_counts = X['cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            st.markdown("#### Cluster Summary")
            st.dataframe(cluster_counts, width='stretch')
        else:
            info_box("Not enough data points for clustering.")


        # -------------------------------------------------------
        # TAB 4: COMBINED SUMMARY & TRENDS
        # -------------------------------------------------------
        with tab4:
            st.markdown("### 📈 Combined Summary & Forecast Trends")

            if not anomalies.empty:
                count_df = anomalies['metric'].value_counts().reset_index()
                count_df.columns = ['Metric', 'Count']

                # Bar chart
                bar_fig = px.bar(
                    count_df,
                    x='Metric', y='Count', color='Metric',
                    title="Anomaly Count per Metric",
                    color_discrete_sequence=["#1E90FF", '#FF4500', '#32CD32']
                )
                bar_fig.update_layout(
                    paper_bgcolor="#C49BFF",
                    plot_bgcolor="#C49BFF",
                    font=dict(color="#4B0082"),
                    height=400
                )
                st.plotly_chart(bar_fig, width='stretch')

                # Pie chart
                pie_fig = px.pie(
                    count_df,
                    names='Metric', values='Count',
                    title="Proportion of Anomalies per Metric",
                    color_discrete_sequence=['#1E90FF', '#FF4500', '#32CD32']
                )
                pie_fig.update_traces(textinfo='percent+label')
                pie_fig.update_layout(paper_bgcolor="#C49BFF", font=dict(color="#4B0082"), height=400)
                st.plotly_chart(pie_fig, width='stretch')
            else:
                success_box("🎉 No anomalies found — your fitness trends are consistent!")

            # Add Prophet-based trend visualization
            st.markdown("#### Health Trends Over Time")
            fig_trends = plot_trends(df, models)
            st.plotly_chart(fig_trends, use_container_width=True)
