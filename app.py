import streamlit as st
from pipeline.preprocessor import FitnessDataPreprocessor

def main():
        st.set_page_config(page_title="FitPulse Preprocessor", page_icon="💓", layout="wide")
        st.title("💓 FitPulse - Data Collection & Preprocessing")
        st.markdown("**Milestone 1: Upload → Validate → Normalize → Preview**")

        if 'preprocessor' not in st.session_state:
            st.session_state.preprocessor = FitnessDataPreprocessor()

        # Sidebar controls
        st.sidebar.header("⚙️ Settings")
        target_frequency = st.sidebar.selectbox(
            "C: Target Frequency:", options=['1min', '5min', '15min', '30min', '1hour'], index=0
        )

        fill_method = st.sidebar.selectbox(
            "C: Missing Value Fill Method:",
            options=['interpolate', 'forward_fill', 'backward_fill', 'zero', 'drop'], index=0
        )

        st.sidebar.markdown("""---
**Missing value strategies:**

- **interpolate:** linear interpolation for numeric columns
- **forward_fill:** propagate last known value
- **backward_fill:** use next known value
- **zero:** fill with 0
- **drop:** drop rows with missing values
""")

        st.header("📁 Upload your fitness CSV / JSON files")
        uploaded_files = st.file_uploader(
            "Choose CSV or JSON files", type=['csv','json'],
            accept_multiple_files=True
        )

        if st.button("🚀 Run Preprocessing Pipeline", type='primary'):
            if not uploaded_files:
                st.error("❌ Please upload at least one CSV or JSON file.")
            else:
                with st.spinner("Running pipeline..."):
                    processed = st.session_state.preprocessor.run_complete_pipeline(
                        uploaded_files=uploaded_files,
                        target_frequency=target_frequency,
                        fill_method=fill_method
                    )
                    if processed:
                        st.success("✅ Preprocessing complete!")
                    else:
                        st.error("❌ Pipeline failed. Check logs above.")

        st.markdown("---")
        st.session_state.preprocessor.create_data_preview_interface()

if __name__ == '__main__':
        main()
