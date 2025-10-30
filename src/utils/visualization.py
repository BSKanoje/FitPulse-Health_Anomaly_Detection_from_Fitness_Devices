import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ----------------------------
# Interactive Plotly functions: Light purple background, distinct line colors
# ----------------------------
APP_BG_COLOR = "#E6E6FA"  # Light purple
FONT_COLOR = "#4B0082"    # Dark purple
HOVER_BG_COLOR = "#F0E6FA"
HOVER_FONT_COLOR = "#4B0082"

# Line colors per metric
LINE_COLORS = {
    'heart_rate': '#FF4500',   # Red
    'steps': '#32CD32',        # Green
    'sleep_duration': '#1E90FF'  # Blue
}
PREDICTED_COLOR = '#9370DB'  # Purple for predicted lines

# ----------------------------
# ANOMALY PLOTS
# ----------------------------
def plot_anomalies(df, anomalies):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=('Heart Rate', 'Steps', 'Sleep Duration'),
        vertical_spacing=0.05
    )

    metrics = ['heart_rate', 'steps', 'sleep_duration']
    anomaly_dict = {row['metric']: row['timestamp'] for _, row in pd.DataFrame(anomalies).iterrows()} if len(anomalies) > 0 else {}

    for i, metric in enumerate(metrics, 1):
        # Main metric line
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[metric], mode='lines', name=f"{metric} Actual",
                line=dict(color=LINE_COLORS[metric], width=2),
                hovertemplate=f"<b>{metric} Actual</b>: %{{y}}<br>Time: %{{x}}<extra></extra>"
            ),
            row=i, col=1
        )

        # Anomaly points
        anom_ts = [ts for m, ts in anomaly_dict.items() if m == metric]
        if anom_ts:
            anom_vals = [df.loc[ts, metric] for ts in anom_ts]
            fig.add_trace(
                go.Scatter(
                    x=anom_ts, y=anom_vals, mode='markers+text',
                    marker=dict(color='black', size=10),
                    name=f"{metric} Anomalies",
                    text=[f"{v:.1f}" for v in anom_vals],
                    textposition="top center",
                    hovertemplate=f"<b>{metric} Anomaly</b>: %{{y}}<br>Time: %{{x}}<extra></extra>"
                ),
                row=i, col=1
            )

    fig.update_layout(
        height=900,
        title_text="Anomaly Detection Results",
        title_font=dict(color=FONT_COLOR, size=24),
        showlegend=True,
        legend=dict(font=dict(color=FONT_COLOR), bgcolor=APP_BG_COLOR),
        plot_bgcolor=APP_BG_COLOR,
        paper_bgcolor=APP_BG_COLOR,
        font=dict(color=FONT_COLOR),
        hoverlabel=dict(bgcolor=HOVER_BG_COLOR, font_size=14, font_color=HOVER_FONT_COLOR)
    )

    # Subplot titles
    for ann in fig.layout.annotations:
        ann.font.color = FONT_COLOR

    fig.update_xaxes(title_text="Time", row=3, col=1, color=FONT_COLOR)
    fig.update_yaxes(color=FONT_COLOR)
    return fig

# ----------------------------
# TREND PLOTS (Actual vs Predicted)
# ----------------------------
def plot_trends(df, models):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=('HR Trend', 'Steps Trend', 'Sleep Trend')
    )

    metrics = ['heart_rate', 'steps', 'sleep_duration']

    for i, metric in enumerate(metrics, 1):
        _, forecast, _ = models.get(metric, (None, df[[metric]], None))

        # ✅ Actual line
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[metric],
                mode='lines+markers',
                name=f"{metric.capitalize()} Actual",
                line=dict(color=LINE_COLORS[metric], width=2),
                hovertemplate=f"<b>{metric} Actual</b>: %{{y}}<br>Time: %{{x}}<extra></extra>"
            ),
            row=i, col=1
        )

        # ✅ Predicted line (bright visible white + glow)
        if forecast is not None and 'yhat' in forecast:
            # Soft white glow for visibility
            forecast['ds'] = pd.to_datetime(forecast['ds'])

            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'], y=forecast['yhat'],
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0.4)', width=10),
                    name=f"{metric.capitalize()} Predicted Glow",
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=i, col=1
            )
            # Main white dashed predicted line
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'], y=forecast['yhat'],
                    mode='lines',
                    name=f"{metric.capitalize()} Predicted",
                    line=dict(color='#BA55D3', dash='dot', width=3),
                    hovertemplate=f"<b>{metric} Predicted</b>: %{{y}}<br>Time: %{{x}}<extra></extra>"
                ),
                row=i, col=1
            )
            print(metric, forecast.head())


    # ✅ Layout styling
    fig.update_layout(
        height=900,
        title_text="📈 Time Series Trends (Actual vs Predicted)",
        title_font=dict(color=FONT_COLOR, size=24),
        showlegend=True,
        legend=dict(font=dict(color=FONT_COLOR), bgcolor=APP_BG_COLOR),
        plot_bgcolor=APP_BG_COLOR,
        paper_bgcolor=APP_BG_COLOR,
        font=dict(color=FONT_COLOR),
        hoverlabel=dict(bgcolor=HOVER_BG_COLOR, font_size=14, font_color=HOVER_FONT_COLOR)
    )

    for ann in fig.layout.annotations:
        ann.font.color = FONT_COLOR

    fig.update_xaxes(color=FONT_COLOR, title_text="Time", row=3, col=1)
    fig.update_yaxes(color=FONT_COLOR)
    return fig
