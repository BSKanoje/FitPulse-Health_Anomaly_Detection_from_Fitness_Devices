import pandas as pd
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
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
    """
    Export CSV and PDF report inside a specified folder.
    Marks anomalies and provides sample alerts.
    """
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Full path for files
    csv_path = os.path.join(folder, f"{filename}.csv")
    pdf_path = os.path.join(folder, f"{filename}.pdf")

    # Mark anomalies in full DF
    full_df = df.copy()
    full_df['is_anomaly'] = False
    if not anomalies.empty:
        for _, row in anomalies.iterrows():
            if row['timestamp'] in full_df.index:
                full_df.loc[row['timestamp'], 'is_anomaly'] = True

    # Export CSV
    full_df.reset_index().to_csv(csv_path, index=False)

    # Create PDF report
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("FitPulse Health Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Data Period: {df.index.min()} to {df.index.max()}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Summary table
    anomaly_df = pd.DataFrame(anomalies) if not anomalies.empty else pd.DataFrame()
    alert_samples = {k: v[0] if v else 'No alerts' for k, v in alerts.items()}
    data = [
        ['Metric', 'Anomalies Found', 'Sample Alert'],
        ['Heart Rate', len(anomaly_df[anomaly_df['metric'] == 'heart_rate']) if not anomaly_df.empty else 0, alert_samples.get('heart_rate', 'No alerts')],
        ['Steps', len(anomaly_df[anomaly_df['metric'] == 'steps']) if not anomaly_df.empty else 0, alert_samples.get('steps', 'No alerts')],
        ['Sleep Duration', len(anomaly_df[anomaly_df['metric'] == 'sleep_duration']) if not anomaly_df.empty else 0, alert_samples.get('sleep_duration', 'No alerts')]
    ]

    from reportlab.platypus import TableStyle
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

    # Detailed anomalies (top 10)
    if not anomalies.empty:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Detected Anomalies:", styles['Heading2']))
        for _, row in anomaly_df.head(10).iterrows():
            story.append(Paragraph(f"{row['timestamp']}: {row['metric']} = {row['value']} ({row['type']})", styles['Normal']))

    # Build PDF
    doc.build(story)

    # Feedback for Streamlit
    if 'st' in globals():
        import streamlit as st
        st.success(f"Reports exported: {csv_path} and {pdf_path}")
    else:
        print(f"Reports exported: {csv_path} and {pdf_path}")
