
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ============================
# FitPulse Theme Colors
# ============================
APP_BG_COLOR = "#E6E6FA"     # Light lavender
FONT_COLOR = "#4B0082"       # Dark purple
HOVER_BG_COLOR = "#F0E6FA"
HOVER_FONT_COLOR = "#4B0082"

LINE_COLORS = {
    'heart_rate': '#FF4500',      # Red
    'steps': '#32CD32',           # Green
    'sleep_duration': '#1E90FF'   # Blue
}

PREDICTED_COLOR = '#BA55D3'  # Purple dashed line
GLOW_COLOR = "rgba(255,255,255,0.4)"

# ===========================================================
# ✅ 1. RULE-BASED + MODEL-BASED ANOMALY PLOTS (Clean Version)
# ===========================================================
def plot_anomalies(df, anomalies):
    """
    Plots actual signals + anomaly markers for all metrics.
    Timestamp on X-axis (MOST IMPORTANT FIX)
    """

    # Ensure anomalies is DataFrame
    anomalies = pd.DataFrame(anomalies) if not isinstance(anomalies, pd.DataFrame) else anomalies

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Heart Rate', 'Steps', 'Sleep Duration'),
        vertical_spacing=0.07
    )

    metrics = ['heart_rate', 'steps', 'sleep_duration']

    for i, metric in enumerate(metrics, 1):

        # ✅ Actual line (always use timestamp column)
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[metric],
                mode='lines+markers',
                name=f"{metric} Actual",
                line=dict(color=LINE_COLORS[metric], width=2),
                hovertemplate=f"<b>{metric}</b>: %{{y}}<br>Time: %{{x}}<extra></extra>"
            ),
            row=i, col=1
        )

        # ✅ Filter anomalies for this metric
        if not anomalies.empty:
            metric_anoms = anomalies[anomalies["metric"] == metric]

            if not metric_anoms.empty:
                fig.add_trace(
                    go.Scatter(
                        x=metric_anoms["timestamp"],
                        y=metric_anoms["value"],
                        mode='markers+text',
                        marker=dict(color="black", size=10),
                        text=[f"{v:.1f}" for v in metric_anoms["value"]],
                        textposition="top center",
                        name=f"{metric} Anomaly",
                        hovertemplate=f"<b>{metric} Anomaly</b>: %{{y}}<br>Time: %{{x}}<extra></extra>"
                    ),
                    row=i, col=1
                )

    # ✅ Layout styling
    fig.update_layout(
        height=900,
        title="Anomaly Detection Results",
        title_font=dict(color=FONT_COLOR, size=24),
        paper_bgcolor=APP_BG_COLOR,
        plot_bgcolor=APP_BG_COLOR,
        font=dict(color=FONT_COLOR),
        showlegend=True,
        hoverlabel=dict(
            bgcolor=HOVER_BG_COLOR,
            font_size=14,
            font_color=HOVER_FONT_COLOR
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # ✅ Fix subplot title colors
    for ann in fig.layout.annotations:
        ann.font.color = FONT_COLOR

    fig.update_xaxes(color=FONT_COLOR)
    fig.update_yaxes(color=FONT_COLOR)

    return fig


# ===========================================================
# ✅ 2. ACTUAL vs PROPHET PREDICTED TRENDS (Perfect Alignment)
# ===========================================================
def plot_trends(df, models):
    """
    Plots Actual vs Predicted using Prophet.
    Ensures both lines use timestamp axis.
    """
    if 'timestamp' not in df.columns:
        df = df.reset_index()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Heart Rate Trend', 'Steps Trend', 'Sleep Trend'),
        vertical_spacing=0.07
    )

    metrics = ['heart_rate', 'steps', 'sleep_duration']

    for i, metric in enumerate(metrics, 1):

        # ✅ Actual data line
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df[metric],
                mode='lines+markers',
                name=f"{metric} Actual",
                line=dict(color=LINE_COLORS[metric], width=2)
            ),
            row=i, col=1
        )

        # ✅ Prophet forecast line
        model, forecast, _ = models.get(metric, (None, None, None))

        if forecast is not None and "yhat" in forecast:
            forecast["ds"] = pd.to_datetime(forecast["ds"])

            # Glow Layer
            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    mode='lines',
                    line=dict(color=GLOW_COLOR, width=10),
                    name=f"{metric} Forecast Glow",
                    showlegend=False
                ),
                row=i, col=1
            )

            # Main predicted line
            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    mode='lines',
                    line=dict(color=PREDICTED_COLOR, dash="dot", width=3),
                    name=f"{metric} Predicted"
                ),
                row=i, col=1
            )

    # ✅ Layout Styling
    fig.update_layout(
        height=900,
        title="📈 Actual vs Predicted Trends",
        title_font=dict(color=FONT_COLOR, size=24),
        paper_bgcolor=APP_BG_COLOR,
        plot_bgcolor=APP_BG_COLOR,
        font=dict(color=FONT_COLOR),
        legend=dict(
            font=dict(color="black"),   # <-- FIX
            title_font=dict(color="black")
        ),
        hoverlabel=dict(
            bgcolor=HOVER_BG_COLOR,
            font_size=14,
            font_color=HOVER_FONT_COLOR
        ),
        showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Fix title colors
    for ann in fig.layout.annotations:
        ann.font.color = FONT_COLOR

    fig.update_xaxes(color=FONT_COLOR, title="Time")
    fig.update_yaxes(color=FONT_COLOR)

    return fig
