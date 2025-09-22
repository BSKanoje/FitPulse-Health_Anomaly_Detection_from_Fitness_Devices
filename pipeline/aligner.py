import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict

class TimeAligner:
    def __init__(self):
        self.supported_frequencies = {'1min':'1T','5min':'5T','15min':'15T','30min':'30T','1hour':'1H'}

    def align_and_resample(self, df: pd.DataFrame, data_type: str, target_frequency: str='1min', fill_method: str='interpolate') -> Tuple[pd.DataFrame, Dict]:
        report = {'original_frequency':None,'target_frequency':target_frequency,'original_rows':len(df),'resampled_rows':0,'gaps_filled':0,'method_used':fill_method,'success':False}
        try:
            if 'timestamp' not in df.columns:
                raise ValueError('No timestamp column found')
            df_indexed = df.set_index('timestamp')
            report['original_frequency'] = self._detect_frequency(df_indexed)

            if target_frequency not in self.supported_frequencies:
                raise ValueError(f'Unsupported frequency: {target_frequency}')
            freq = self.supported_frequencies[target_frequency]

            df_resampled = self._resample_by_type(df_indexed, data_type, freq)
            df_filled, gaps = self._fill_missing_after_resample(df_resampled, data_type, fill_method)
            df_final = df_filled.reset_index()
            report['resampled_rows'] = len(df_final)
            report['gaps_filled'] = gaps
            report['success'] = True
            return df_final, report
        except Exception as e:
            report['error'] = str(e)
            return df, report

    def _detect_frequency(self, df_indexed: pd.DataFrame) -> str:
        try:
            if len(df_indexed) < 2:
                return 'insufficient_data'
            diffs = df_indexed.index.to_series().diff().dropna()
            mode = diffs.mode()
            if len(mode)==0:
                return 'irregular'
            mins = mode.iloc[0].total_seconds()/60
            if mins < 1: return 'sub_minute'
            mapping = {1:'1min',5:'5min',15:'15min',30:'30min',60:'1hour'}
            return mapping.get(int(mins), f"{mins:.1f}min")
        except:
            return 'unknown'

    def _resample_by_type(self, df_indexed: pd.DataFrame, data_type: str, freq_str: str) -> pd.DataFrame:
        res = {}
        for col in df_indexed.columns:
            if col.endswith('_outlier'):
                res[col] = df_indexed[col].resample(freq_str).max()
            elif col == 'heart_rate':
                res[col] = df_indexed[col].resample(freq_str).mean()
            elif col == 'step_count':
                res[col] = df_indexed[col].resample(freq_str).sum()
            elif col == 'duration_minutes':
                res[col] = df_indexed[col].resample(freq_str).sum()
            elif col == 'sleep_stage':
                res[col] = df_indexed[col].resample(freq_str).agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else np.nan)
            else:
                if pd.api.types.is_numeric_dtype(df_indexed[col]):
                    res[col] = df_indexed[col].resample(freq_str).mean()
                else:
                    res[col] = df_indexed[col].resample(freq_str).first()
        return pd.DataFrame(res)

    def _fill_missing_after_resample(self, df: pd.DataFrame, data_type: str, fill_method: str):
        initial_missing = int(df.isnull().sum().sum())
        if fill_method == 'interpolate':
            num_cols = df.select_dtypes(include=['number']).columns
            for col in num_cols:
                if not col.endswith('_outlier'):
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
            cat_cols = df.select_dtypes(exclude=['number']).columns
            for col in cat_cols:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        elif fill_method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif fill_method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif fill_method == 'zero':
            df = df.fillna(0)
        elif fill_method == 'drop':
            df = df.dropna()
        final_missing = int(df.isnull().sum().sum())
        gaps_filled = initial_missing - final_missing
        return df, gaps_filled
    
    def generate_alignment_report(self, align_report):
        # convert align_report to a string or desired format
        return str(align_report)
