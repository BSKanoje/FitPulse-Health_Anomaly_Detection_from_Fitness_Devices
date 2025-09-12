# FitPulse - Health Anomaly Detection from Fitness Devices

FitPulse is a **real-time health monitoring system** that collects data from fitness devices, preprocesses it, and applies **time-series analysis** to detect anomalies such as sudden heart rate spikes, irregular patterns, or unusual activity trends. It generates **personalized health alerts**, visualizes anomalies on an **interactive dashboard**, and allows users to **download detailed health reports** for further insights.

---

## 🚀 Objectives
1. Detect anomalies in health data
2. Generate personalized health alerts
3. Support real-time interactive dashboards for users and doctors
4. Allow downloading of detailed health reports

---

## 📊 Data Sources
FitPulse uses the following metrics for anomaly detection:
- **Heart Rate**
- **Sleep Duration**
- **Step Count**

Data should be collected in **CSV** or **JSON** format.

---

## 🏗️ Project Modules
1. **Data Collection & Preprocessing**  
   - Input formats: CSV / JSON  
   - Cleaning, normalization, and handling missing values

2. **Feature Extraction & Modelling**  
   - Extract features from time-series using **TSFresh**
   - Model temporal dependencies using **Facebook Prophet**

3. **Anomaly Detection & Visualization**  
   - Apply clustering algorithms (**K-Means, DBSCAN**) for anomaly grouping
   - Visualize results using **Matplotlib**

4. **Dashboard for Insights**  
   - Real-time monitoring with **Streamlit**
   - Generate alerts & download health reports

---

## 🛠️ Tools & Technologies
- **Language**: Python
- **Libraries**:
  - `Pandas`, `NumPy` → data handling & preprocessing
  - `Matplotlib` → visualization
  - `TSFresh` → feature extraction
  - `Facebook Prophet` → time-series modelling
  - `K-Means`, `DBSCAN` → clustering
- **Dashboard**: Streamlit

---

## ⏱️ Time Series in Anomaly Detection
A **time series** is data collected over time at regular intervals (e.g., heart rate readings every minute).

Time-series analysis helps detect anomalies by comparing values against trends and seasonal patterns:

### Why It Matters:
- A value may be abnormal only compared to **previous trends** (e.g., sudden HR spike during rest)
- **Patterns** (daily, weekly, seasonal) provide context — what’s normal at night might be abnormal in the day

### Main Types of Time Series Analysis
| Type       | What It Shows              | Fitness Example                  | Importance |
|------------|----------------------------|----------------------------------|------------|
| **Trend**  | Long-term direction         | Avg steps increasing every month | Detect improvement or decline |
| **Seasonality** | Regular repeating pattern | HR rises every morning after waking | Detect missing/unexpected cycles |
| **Residual** | Random variation after removing trend & seasonality | Sudden HR spike at night | Detect anomalies |

---

## 🔍 Types of Anomalies
| Type        | Definition                            | Fitness Example |
|-------------|---------------------------------------|-----------------|
| **Point**   | Single value far from normal          | One sudden HR spike at rest |
| **Contextual** | Value abnormal only in specific context | HR 120 bpm while sleeping |
| **Collective** | Sequence/group of values forms anomaly | Slightly high HR all night |

---

## 📦 Installation & Setup
```bash
# Clone the repository
git clone https://github.com/your-username/FitPulse.git
cd FitPulse

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py
```

---

## 📑 Future Enhancements
- Integration with wearable APIs (Fitbit, Apple Watch, Garmin)
- More advanced anomaly detection using deep learning (LSTMs, Autoencoders)
- Cloud-based storage & doctor-patient collaboration features

---

## 👨‍⚕️ Use Cases
- **Individuals**: Track personal health & get real-time alerts
- **Doctors**: Monitor patients remotely with anomaly reports
- **Researchers**: Study long-term patterns & detect unusual trends

---

## 📝 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
