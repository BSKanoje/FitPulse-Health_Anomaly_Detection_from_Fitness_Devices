import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict
from modules.uploader import FitnessDataUploader
from modules.validator import FitnessDataValidator
from modules.aligner import TimeAligner

class FitnessDataPreprocessor:
    def __init__(self):
        self.uploader = FitnessDataUploader()
        self.validator = FitnessDataValidator()
        self.aligner = TimeAligner()
        self.processing_log = []
        self.processed_data = {}
        self.reports = {}

    def run_complete_pipeline(self, uploaded_files=None, target_frequency='1min', fill_method='interpolate') -> Dict[str,pd.DataFrame]:
        """Run the full preprocessing pipeline: upload → validate → align → summarize."""
        # Clear previous logs and processed data
        self.processing_log = []
        self.processed_data = {}
        self.reports = {}

        st.header("🔄 Data Preprocessing Pipeline")
        self.log_step("🔵 COMPONENT A: Starting data upload and loading...")

        if not uploaded_files:
            st.error('No files provided. Please upload CSV/JSON files.')
            return {}

        raw_data = self.uploader.create_upload_interface(uploaded_files)
        if not raw_data:
            st.error('No valid data loaded from uploaded files.')
            return {}

        # Component B: Validation
        self.log_step('🟡 COMPONENT B: Validating and cleaning data...')
        validated = {}
        for data_type, df in raw_data.items():
            cleaned, val_report = self.validator.validate_and_clean_data(df, data_type)
            validated[data_type] = cleaned
            self.reports[f"{data_type}_validation"] = val_report
            st.subheader(f"📋 {data_type.title()} Validation Results")
            st.text(self.validator.generate_validation_report(val_report))

        # Component C: Time Alignment
        self.log_step('🟢 COMPONENT C: Aligning timestamps and resampling data...')
        aligned = {}
        for data_type, df in validated.items():
            aligned_df, align_report = self.aligner.align_and_resample(df, data_type, target_frequency, fill_method)
            aligned[data_type] = aligned_df
            self.reports[f"{data_type}_alignment"] = align_report
            st.subheader(f"⏰ {data_type.title()} Time Alignment Results")
            st.text(self.aligner.generate_alignment_report(align_report))

        # Integration complete
        self.log_step('✅ INTEGRATION: Final data quality checks and pipeline completion...')
        self.processed_data = aligned
        self._generate_processing_summary()
        return aligned

    def log_step(self, message: str):
        """Log a processing step with timestamp."""
        from datetime import datetime
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        entry = f"[{ts}] {message}"
        self.processing_log.append(entry)
        st.info(entry)

    def create_data_preview_interface(self):
        """Display processed data metrics, sample, and interactive visualizations."""
        if not self.processed_data:
            st.warning('No processed data available. Run the pipeline first.')
            return

        st.header('📊 Processed Data Preview')
        data_type = st.selectbox(
            'Select file to preview:', 
            list(self.processed_data.keys()), key='data_type_preview'
        )

        if data_type in self.processed_data:
            df = self.processed_data[data_type]
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric('📁 A: Total Records', len(df))
            with col2:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric('🔧 B: Data Quality', f"{100-missing_pct:.1f}%")
            with col3:
                try:
                    time_span_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
                except:
                    time_span_hours = 0
                st.metric('⏰ C: Time Span', f"{time_span_hours:.1f}h")
            with col4:
                st.metric('🔗 Integration', '✅ Complete')

            st.subheader('Data Sample')
            st.dataframe(df.head(50), use_container_width=True)

            # Visualization
            numeric_columns = df.select_dtypes(include=['number']).columns
            numeric_columns = [c for c in numeric_columns if not c.endswith('_outlier')]

            if numeric_columns:
                selected = st.selectbox('Select metric to visualize', numeric_columns, key='metric_preview')
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df[selected], mode='lines+markers', name=selected
                ))

                # Plot outliers if flagged
                out_col = selected + '_outlier'
                if out_col in df.columns:
                    out_df = df[df[out_col]==True]
                    if not out_df.empty:
                        fig.add_trace(go.Scatter(
                            x=out_df['timestamp'], y=out_df[selected],
                            mode='markers',
                            name='Outliers',
                            marker=dict(symbol='x', size=10, color='red')
                        ))

                fig.update_layout(
                    title=f"{selected} over time",
                    xaxis_title='timestamp',
                    yaxis_title=selected,
                    height=480
                )
                st.plotly_chart(fig, use_container_width=True)

    def _generate_processing_summary(self):
        """Display a high-level summary of pipeline components and log."""
        st.header('📝 Complete Pipeline Summary')

        c1, c2, c3 = st.columns(3)
        with c1:
            st.success('🔵 Component A: File Upload')
            st.write('✅ Multi-format file support')
            st.write('✅ Data type auto-detection')
        with c2:
            st.success('🟡 Component B: Data Validation')
            st.write('✅ Missing value handling')
            st.write('✅ Outlier detection')
        with c3:
            st.success('🟢 Component C: Time Alignment')
            st.write('✅ Timestamp normalization')
            st.write('✅ Frequency resampling')

        st.subheader('Processing Log')
        for entry in self.processing_log:
            st.text(entry)


# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# from typing import Dict
# from modules.uploader import FitnessDataUploader
# from modules.validator import FitnessDataValidator
# from modules.aligner import TimeAligner

# class FitnessDataPreprocessor:
#     def __init__(self):
#         self.uploader = FitnessDataUploader()
#         self.validator = FitnessDataValidator()
#         self.aligner = TimeAligner()
#         self.processing_log = []
#         self.processed_data = {}
#         self.reports = {}

#     def run_complete_pipeline(self, uploaded_files=None, target_frequency='1min', fill_method='interpolate') -> Dict[str,pd.DataFrame]:
#         st.header("🔄 Data Preprocessing Pipeline")
#         self.log_step("🔵 COMPONENT A: Starting data upload and loading...")
#         if not uploaded_files:
#             st.error('No files provided. Please upload CSV/JSON files.')
#             return {}
#         raw_data = self.uploader.create_upload_interface(uploaded_files)
#         if not raw_data:
#             st.error('No valid data loaded from uploaded files.')
#             return {}
#         self.log_step('🟡 COMPONENT B: Validating and cleaning data...')
#         validated = {}
#         for data_type, df in raw_data.items():
#             cleaned, val_report = self.validator.validate_and_clean_data(df, data_type)
#             validated[data_type] = cleaned
#             self.reports[f"{data_type}_validation"] = val_report
#             st.subheader(f"📋 {data_type.title()} Validation Results")
#             st.text(self.validator.generate_validation_report(val_report))
#         self.log_step('🟢 COMPONENT C: Aligning timestamps and resampling data...')
#         aligned = {}
#         for data_type, df in validated.items():
#             aligned_df, align_report = self.aligner.align_and_resample(df, data_type, target_frequency, fill_method)
#             aligned[data_type] = aligned_df
#             self.reports[f"{data_type}_alignment"] = align_report
#             st.subheader(f"⏰ {data_type.title()} Time Alignment Results")
#             st.text(self.aligner.generate_alignment_report(align_report))
#         self.log_step('✅ INTEGRATION: Final data quality checks and pipeline completion...')
#         self.processed_data = aligned
#         self._generate_processing_summary()
#         return aligned

#     def log_step(self, message: str):
#         from datetime import datetime
#         ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
#         entry = f"[{ts}] {message}"
#         self.processing_log.append(entry)
#         st.info(entry)

#     def create_data_preview_interface(self):
#         if not self.processed_data:
#             st.warning('No processed data available. Run the pipeline first.')
#             return
#         st.header('📊 Processed Data Preview (A+B+C Results)')
#         data_type = st.selectbox('Select data type to preview:', list(self.processed_data.keys()))
#         if data_type in self.processed_data:
#             df = self.processed_data[data_type]
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric('📁 A: Total Records', len(df))
#             with col2:
#                 missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
#                 st.metric('🔧 B: Data Quality', f"{100-missing_pct:.1f}%")
#             with col3:
#                 try:
#                     time_span_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
#                 except:
#                     time_span_hours = 0
#                 st.metric('⏰ C: Time Span', f"{time_span_hours:.1f}h")
#             with col4:
#                 st.metric('🔗 Integration', '✅ Complete')
#             st.subheader('Data Sample')
#             st.dataframe(df.head(50), use_container_width=True)
#             # visualization
#             numeric_columns = df.select_dtypes(include=['number']).columns
#             numeric_columns = [c for c in numeric_columns if not c.endswith('_outlier')]
#             if numeric_columns:
#                 selected = st.selectbox('Select metric to visualize', numeric_columns)
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(x=df['timestamp'], y=df[selected], mode='lines+markers', name=selected))
#                 out_col = selected + '_outlier'
#                 if out_col in df.columns:
#                     out_df = df[df[out_col]==True]
#                     if not out_df.empty:
#                         fig.add_trace(go.Scatter(x=out_df['timestamp'], y=out_df[selected], mode='markers', name='Outliers', marker=dict(symbol='x', size=10, color='red')))
#                 fig.update_layout(title=f"{selected} over time", xaxis_title='timestamp', yaxis_title=selected, height=480)
#                 st.plotly_chart(fig, use_container_width=True)

#     def _generate_processing_summary(self):
#         st.header('📝 Complete Pipeline Summary (A+B+C)')
#         st.subheader('🔄 Pipeline Component Integration')
#         c1,c2,c3 = st.columns(3)
#         with c1:
#             st.success('🔵 Component A: File Upload')
#             st.write('✅ Multi-format file support')
#             st.write('✅ Data type auto-detection')
#         with c2:
#             st.success('🟡 Component B: Data Validation')
#             st.write('✅ Missing value handling')
#             st.write('✅ Outlier detection')
#         with c3:
#             st.success('🟢 Component C: Time Alignment')
#             st.write('✅ Timestamp normalization')
#             st.write('✅ Frequency resampling')
#         st.subheader('Processing Log')
#         for entry in self.processing_log:
#             st.text(entry)
