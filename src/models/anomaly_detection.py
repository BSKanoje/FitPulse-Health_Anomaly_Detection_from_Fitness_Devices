import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class AnomalyDetector:
    """
    Detects anomalies in fitness data using rule-based and model-based methods.
    """

    def __init__(self, df):
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={'index': 'timestamp'})
            else:
                raise KeyError("No 'timestamp' column or datetime index found in DataFrame.")
        self.df = df

    # ---------------------------------------------------
    # 1️⃣ RULE-BASED DETECTION
    # ---------------------------------------------------
    def rule_based_detection(self):
        """
        Detect anomalies using fixed thresholds for key metrics.
        Returns a dictionary of anomalies and summary.
        """
        rules = {
            'heart_rate': (50, 110),
            'steps': (5, 120),
            'sleep_duration': (60, 600)
        }

        anomalies = {}
        summary = {}

        for metric, (low, high) in rules.items():
            if metric in self.df.columns:
                mask = (self.df[metric] < low) | (self.df[metric] > high)
                anomaly_indices = self.df.index[mask].tolist()
                anomalies[metric] = anomaly_indices
                summary[metric] = len(anomaly_indices)

        summary['total_rule_based'] = sum(summary.values())
        return anomalies, summary

    # ---------------------------------------------------
    # 2️⃣ MODEL-BASED DETECTION (K-MEANS CLUSTERING)
    # ---------------------------------------------------
    def cluster_based_detection(self):
        """
        Detect outlier clusters using KMeans (model-based approach).
        Returns DataFrame of outlier rows.
        """
        features = ['heart_rate', 'steps', 'sleep_duration']
        available_features = [f for f in features if f in self.df.columns]
        model_features = self.df[available_features].fillna(0)

        if len(model_features) < 5:
            return pd.DataFrame()  # Not enough data

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(model_features)
        self.df['cluster'] = clusters

        outlier_cluster = pd.Series(clusters).value_counts().idxmin()
        outliers = self.df[self.df['cluster'] == outlier_cluster]

        return outliers

    # ---------------------------------------------------
    # 3️⃣ COMBINED WRAPPER FUNCTION
    # ---------------------------------------------------
    def detect_all(self):
        """
        Run both detection methods and return results + alert messages.
        """
        anomalies, summary = self.rule_based_detection()
        outliers = self.cluster_based_detection()

        alerts = []
        for metric, count in summary.items():
            if metric != 'total_rule_based' and count > 0:
                alerts.append(f"{count} {metric} anomalies detected (Rule-based)")

        if not outliers.empty:
            alerts.append(f"{len(outliers)} cluster-based anomaly points detected")

        total = summary['total_rule_based'] + len(outliers)

        return {
            'anomalies': anomalies,
            'outliers': outliers,
            'alerts': alerts,
            'summary': summary,
            'total_anomalies': total
        }
    
def detect_anomalies(df, features=None, models=None):
    """
    Wrapper function for backward compatibility.
    Runs rule-based and cluster-based anomaly detection using AnomalyDetector.
    """
    detector = AnomalyDetector(df)
    results = detector.detect_all()

    anomalies = []
    for metric, indices in results['anomalies'].items():
        for i in indices:
            try:
                anomalies.append({
                    'metric': metric,
                    'index': i,
                    'timestamp': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i,
                    'value': df.iloc[i][metric]
                })
            except Exception:
                continue


    anomalies_df = pd.DataFrame(anomalies)
    alerts = results['alerts']

    return anomalies_df, alerts
