import pandas as pd
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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

    # Convert anomalies to DataFrame
    anomaly_df = pd.DataFrame(anomalies) if not anomalies.empty else pd.DataFrame()

    # ------------------------------------------------------
    # ✅ Fix alert samples
    # ------------------------------------------------------
    alert_samples = {m: "No alerts" for m in ["heart_rate", "steps", "sleep_duration"]}

    if isinstance(alerts, dict):
        for metric in alert_samples:
            if metric in alerts and alerts[metric]:
                alert_samples[metric] = alerts[metric][0]

    elif isinstance(alerts, list):
        for alert in alerts:
            t = alert.lower()
            if "heart" in t:
                alert_samples["heart_rate"] = alert
            elif "steps" in t:
                alert_samples["steps"] = alert
            elif "sleep" in t:
                alert_samples["sleep_duration"] = alert

    # ------------------------------------------------------
    # ✅ PDF Setup with FitPulse Purple Theme
    # ------------------------------------------------------
    PURPLE = colors.HexColor("#4B0082")
    LAVENDER = colors.HexColor("#E6E6FA")
    LIGHT_PURPLE = colors.HexColor("#D8BFD8")

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom Title Style
    styles.add(ParagraphStyle(
        name="PurpleTitle",
        fontSize=22,
        leading=26,
        alignment=1,
        textColor=PURPLE,
        spaceAfter=14,
        spaceBefore=10
    ))

    # Section heading
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontSize=16,
        leading=20,
        textColor=PURPLE,
        spaceAfter=10,
        spaceBefore=20
    ))

    # Normal text
    styles.add(ParagraphStyle(
        name="NormalText",
        fontSize=12,
        textColor=colors.black,
        leading=16
    ))

    story = []

    # ------------------------------------------------------
    # ✅ Header
    # ------------------------------------------------------
    story.append(Paragraph("FitPulse Health Anomaly Report", styles['PurpleTitle']))
    story.append(Paragraph(
        f"Data Period: <b>{df.index.min()}</b> to <b>{df.index.max()}</b>",
        styles['NormalText']
    ))
    story.append(Spacer(1, 12))

    # Divider line
    story.append(Paragraph("<br/><hr width='100%' color='#4B0082'/>", styles['NormalText']))

    # ------------------------------------------------------
    # ✅ Summary Table
    # ------------------------------------------------------
    story.append(Paragraph("📊 Summary of Anomalies", styles['SectionHeader']))

    data = [
        ['Metric', 'Anomalies Found', 'Sample Alert'],
        ['Heart Rate',
            len(anomaly_df[anomaly_df['metric'] == 'heart_rate']) if not anomaly_df.empty else 0,
            alert_samples['heart_rate']
        ],
        ['Steps',
            len(anomaly_df[anomaly_df['metric'] == 'steps']) if not anomaly_df.empty else 0,
            alert_samples['steps']
        ],
        ['Sleep Duration',
            len(anomaly_df[anomaly_df['metric'] == 'sleep_duration']) if not anomaly_df.empty else 0,
            alert_samples['sleep_duration']
        ]
    ]

    table = Table(data, colWidths=[120, 120, 260])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PURPLE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),

        # Alternating row colors
        ('BACKGROUND', (0, 1), (-1, 1), LAVENDER),
        ('BACKGROUND', (0, 2), (-1, 2), LIGHT_PURPLE),
        ('BACKGROUND', (0, 3), (-1, 3), LAVENDER),

        ('GRID', (0, 0), (-1, -1), 1, PURPLE)
    ]))

    story.append(table)

    # ------------------------------------------------------
    # ✅ Detailed Anomalies (ALL, Not Top 10)
    # ------------------------------------------------------
    story.append(Paragraph("📌 Detailed List of All Anomalies", styles['SectionHeader']))

    if anomaly_df.empty:
        story.append(Paragraph("✅ No anomalies detected.", styles['NormalText']))
    else:
        for _, row in anomaly_df.iterrows():
            story.append(Paragraph(
                f"<b>{row['timestamp']}</b> — <b>{row['metric']}</b>: {row['value']}",
                styles['NormalText']
            ))
            story.append(Spacer(1, 4))

    # Build PDF
    doc.build(story)

    # Success feedback
    if 'st' in globals():
        import streamlit as st
        st.success(f"✅ PDF report exported successfully: {pdf_path}")
    else:
        print(f"✅ PDF report exported successfully: {pdf_path}")
