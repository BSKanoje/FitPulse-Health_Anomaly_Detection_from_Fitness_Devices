import pandas as pd
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os

def load_data(file_path_or_df):
    """
    Load CSV/JSON file or use a DataFrame.
    Ensures 'timestamp' exists and sets it as index.
    """
    if isinstance(file_path_or_df, str):  # File path
        if file_path_or_df.endswith('.csv'):
            df = pd.read_csv(file_path_or_df)
        elif file_path_or_df.endswith('.json'):
            with open(file_path_or_df, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
    else:  # Already a DataFrame
        df = file_path_or_df.copy()

    # Ensure 'timestamp' column exists
    if 'timestamp' not in df.columns:
        df.insert(0, 'timestamp', pd.date_range(start='2025-01-01', periods=len(df), freq='H'))

    # Convert to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    return df.sort_index()

def export_report(df, anomalies, alerts, filename='fitpulse_report', folder='report'):

    # Create folder if missing
    if not os.path.exists(folder):
        os.makedirs(folder)

    pdf_path = os.path.join(folder, f"{filename}.pdf")

    # ---- Convert anomalies to DataFrame ----
    anomaly_df = pd.DataFrame(anomalies) if not anomalies.empty else pd.DataFrame()

    # ------------------------------------------------------
    # ✅ FIX: Correct metric-wise alert detection
    # ------------------------------------------------------
    alert_samples = {
        "heart_rate": "No alerts",
        "steps": "No alerts",
        "sleep_duration": "No alerts"
    }

    if isinstance(alerts, dict):
        # Already structured as metric → list
        for metric in alert_samples.keys():
            if metric in alerts and isinstance(alerts[metric], list) and alerts[metric]:
                alert_samples[metric] = alerts[metric][0]

    elif isinstance(alerts, list):
        # Detect metric name from alert text
        for alert in alerts:
            if not isinstance(alert, str):
                continue

            text = alert.lower()

            if "heart_rate" in text or "heart rate" in text:
                alert_samples["heart_rate"] = alert

            elif "steps" in text:
                alert_samples["steps"] = alert

            elif "sleep_duration" in text or "sleep duration" in text or "sleep" in text:
                alert_samples["sleep_duration"] = alert

    # ------------------------------------------------------
    # ✅ PDF Setup
    # ------------------------------------------------------
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Header
    story.append(Paragraph("FitPulse Health Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Data Period: {df.index.min()} to {df.index.max()}", styles['Normal']))
    story.append(Spacer(1, 12))

    # ------------------------------------------------------
    # ✅ Summary Table
    # ------------------------------------------------------
    data = [
        ['Metric', 'Anomalies Found', 'Sample Alert'],
        [
            'Heart Rate',
            len(anomaly_df[anomaly_df['metric'] == 'heart_rate']) if not anomaly_df.empty else 0,
            alert_samples["heart_rate"]
        ],
        [
            'Steps',
            len(anomaly_df[anomaly_df['metric'] == 'steps']) if not anomaly_df.empty else 0,
            alert_samples["steps"]
        ],
        [
            'Sleep Duration',
            len(anomaly_df[anomaly_df['metric'] == 'sleep_duration']) if not anomaly_df.empty else 0,
            alert_samples["sleep_duration"]
        ]
    ]

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(table)

    # ------------------------------------------------------
    # ✅ Detailed Anomalies (Top 10)
    # ------------------------------------------------------
    if not anomaly_df.empty:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Detected Anomalies (Top 10):", styles['Heading2']))

        for _, row in anomaly_df.head(10).iterrows():
            anomaly_type = row['type'] if 'type' in row else "Unknown"

            story.append(Paragraph(
                f"{row['timestamp']}: {row['metric']} = {row['value']} ",
                styles['Normal']
            ))

    # ------------------------------------------------------
    # ✅ Build PDF
    # ------------------------------------------------------
    doc.build(story)

    # Feedback
    if 'st' in globals():
        import streamlit as st
        st.success(f"✅ PDF report exported: {pdf_path}")
    else:
        print(f"✅ PDF report exported: {pdf_path}")
