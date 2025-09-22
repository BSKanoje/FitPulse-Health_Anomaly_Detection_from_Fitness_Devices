import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, List, Dict

class FitnessDataValidator:
    def __init__(self):
        self.validation_rules = {
            'heart_rate': {'min_value': 30, 'max_value': 220},
            'step_count': {'min_value': 0, 'max_value': 100000},
            'duration_minutes': {'min_value': 0, 'max_value': 1440}
        }

    def validate_and_clean_data(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        report = {
            'original_rows': len(df),
            'issues_found': [],
            'rows_removed': 0,
            'missing_values_handled': 0,
            'outliers_flagged': 0
        }
        try:
            dfc = df.copy()
            dfc = self._standardize_columns(dfc)
            dfc, ts_issues = self._clean_timestamps(dfc)
            report['issues_found'].extend(ts_issues)

            dfc, num_issues = self._validate_numeric_columns(dfc, data_type)
            report['issues_found'].extend(num_issues)

            dfc, missing_count = self._handle_missing_values(dfc, data_type)
            report['missing_values_handled'] = missing_count

            dfc, outlier_count = self._detect_outliers(dfc, data_type)
            report['outliers_flagged'] = outlier_count

            initial_len = len(dfc)
            dfc = self._remove_invalid_rows(dfc)
            report['rows_removed'] = initial_len - len(dfc)
            report['final_rows'] = len(dfc)
            report['success'] = True
        except Exception as e:
            report['success'] = False
            report['error'] = str(e)
        return dfc, report

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            'time':'timestamp','date':'timestamp','datetime':'timestamp',
            'hr':'heart_rate','heartrate':'heart_rate','heart rate':'heart_rate',
            'steps':'step_count','step':'step_count','stepcount':'step_count',
            'sleep':'sleep_stage','stage':'sleep_stage','duration':'duration_minutes'
        }
        df = df.rename(columns=col_map)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df

    def _clean_timestamps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        issues = []
        if 'timestamp' not in df.columns:
            issues.append('No timestamp column found')
            return df, issues
        parsed = pd.to_datetime(df['timestamp'], errors='coerce', infer_datetime_format=True)
        failed = parsed.isna().sum()
        if failed>0:
            issues.append(f'Failed to parse {failed} timestamps')
        df['timestamp'] = parsed
        # timezone normalize
        try:
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        except Exception:
            # fallback: ensure tz-aware by using UTC without raising
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC', ambiguous='NaT', nonexistent='shift_forward')
        return df, issues

    def _validate_numeric_columns(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, List[str]]:
        issues = []
        # ensure numeric where expected
        for col in ['heart_rate','step_count','duration_minutes']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col in self.validation_rules:
                    minv = self.validation_rules[col]['min_value'] if 'min_value' in self.validation_rules[col] else None
                    maxv = self.validation_rules[col]['max_value'] if 'max_value' in self.validation_rules[col] else None
                    if minv is not None:
                        negs = (df[col] < minv).sum()
                        if negs>0:
                            issues.append(f'Found {negs} values < {minv} in {col}; clipped/naed')
                            df.loc[df[col] < minv, col] = np.nan
                    if maxv is not None:
                        highs = (df[col] > maxv).sum()
                        if highs>0:
                            issues.append(f'Found {highs} values > {maxv} in {col}; clipped/naed')
                            df.loc[df[col] > maxv, col] = np.nan
        return df, issues

    def _handle_missing_values(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, int]:
        missing = int(df.isnull().sum().sum())
        if missing==0:
            return df, 0
        for col in df.columns:
            if df[col].isnull().sum()>0:
                if col=='timestamp':
                    df = df.dropna(subset=['timestamp'])
                elif col in ['heart_rate','step_count']:
                    # forward fill limited, then linear interpolate
                    df[col] = df[col].fillna(method='ffill', limit=5)
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                elif col=='duration_minutes':
                    df[col] = df[col].fillna(df[col].median())
                elif col=='sleep_stage':
                    df[col] = df[col].fillna(method='ffill')
        return df, missing

    def _detect_outliers(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, int]:
        out_count = 0
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            if col=='timestamp':
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5*iqr
            upper = q3 + 1.5*iqr
            mask = (df[col] < lower) | (df[col] > upper)
            if col+'_outlier' in df.columns:
                df[col+'_outlier'] = df[col+'_outlier'] | mask
            else:
                df[col+'_outlier'] = mask.fillna(False)
            out_count += int(mask.sum())
        return df, out_count

    def _remove_invalid_rows(self, df: pd.DataFrame):
        df = df.dropna(subset=['timestamp'])
        val_cols = [c for c in df.columns if c!='timestamp' and not c.endswith('_outlier')]
        df = df.dropna(subset=val_cols, how='all')
        return df

    def generate_validation_report(self, report: Dict) -> str:
        txt = f"""
Original rows: {report.get('original_rows')}
Final rows: {report.get('final_rows','N/A')}
Rows removed: {report.get('rows_removed',0)}
Missing values handled: {report.get('missing_values_handled',0)}
Outliers flagged: {report.get('outliers_flagged',0)}
"""
        if report.get('issues_found'):
            for it in report['issues_found']:
                txt += f"• {it}\n"
        else:
            txt += "• No issues found\n"
        return txt
