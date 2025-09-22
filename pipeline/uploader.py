import pandas as pd
import streamlit as st
import json

class FitnessDataUploader:
    """Handles file upload and initial data loading for fitness tracker data"""
    def __init__(self):
        self.supported = ['.csv', '.json']

    def create_upload_interface(self, uploaded_files):
        data_dict = {}
        for uploaded_file in uploaded_files:
            name = uploaded_file.name.lower()
            try:
                if name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif name.endswith('.json'):
                    # try reading JSON - handle both list/dict forms
                    try:
                        df = pd.read_json(uploaded_file)
                    except Exception:
                        uploaded_file.seek(0)
                        raw = uploaded_file.read()
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict) and 'data' in parsed:
                            df = pd.DataFrame(parsed['data'])
                        elif isinstance(parsed, dict):
                            df = pd.DataFrame([parsed])
                        elif isinstance(parsed, list):
                            df = pd.DataFrame(parsed)
                        else:
                            raise ValueError('Unsupported JSON structure')
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue

                df['source_file'] = uploaded_file.name
                data_type = self._detect_data_type(uploaded_file.name)
                data_dict.setdefault(data_type, pd.concat([data_dict.get(data_type, pd.DataFrame()), df], ignore_index=True))
                st.success(f"Loaded {uploaded_file.name} as {data_type} ({len(df)} rows)")
            except Exception as e:
                st.error(f"Failed to load {uploaded_file.name}: {e}")
        return data_dict

    def _detect_data_type(self, filename: str) -> str:
        """
        Detects the type of fitness data from the filename.
        If unable to detect, uses a meaningful default key based on the filename.
        """
        fn = filename.lower()
        if 'heart' in fn or 'hr' in fn:
            return 'heart_rate'
        if 'sleep' in fn:
            return 'sleep'
        if 'step' in fn or 'activity' in fn:
            return 'steps'
        
        return filename.rsplit('.', 1)[0]  # e.g., "fitness_data.csv" -> "fitness_data"
