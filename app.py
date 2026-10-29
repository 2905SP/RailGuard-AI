"""
RailGuard AI - Railway Track Failure Risk Prediction Dashboard
==============================================================
An interactive, educational machine learning web application built with Streamlit.
Provides track condition analysis, ML-based failure risk inference, exploratory
data analytics, and model evaluation metrics.

Theme: Modern Railway Dark-Navy & Signal Accents.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RailGuard AI | Track Failure Risk Prediction",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Railway Dark-Navy CSS

CUSTOM_CSS = """
<style>
    /* Main application background and default text */
    .stApp {
        background: linear-gradient(
            135deg,
            #0a1118 0%,
            #0d1b2a 50%,
            #102030 100%
        );
        color: #f1f5f9 !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main content text */
    .stApp p,
    .stApp li,
    .stApp label,
    .stApp .stMarkdown,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"] {
        color: #e2e8f0 !important;
    }

    /* Headings */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0c1520;
        border-right: 1px solid #2b4c7e;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }

    /* Header banner */
    .rail-header {
        background: linear-gradient(90deg, #162a45 0%, #1f3b60 100%);
        border-left: 6px solid #ff7b00;
        padding: 22px 25px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.35);
    }

    .rail-header h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 0.3px;
    }

    .rail-header p {
        color: #e2e8f0 !important;
        margin: 8px 0 0;
        font-size: 15px;
        line-height: 1.6;
    }

    /* Metric cards */
    .metric-card {
        background: #172a40;
        border: 1px solid #355477;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        min-height: 110px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    }

    .metric-card .num {
        font-size: 28px;
        font-weight: 800;
        color: #7dd3fc !important;
        margin-top: 8px;
    }

    .metric-card .label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #e2e8f0 !important;
        letter-spacing: 0.5px;
    }

    /* Safe result badge */
    .status-badge-safe {
        background: linear-gradient(90deg, #14532d, #166534);
        border-left: 6px solid #4ade80;
        color: #ffffff !important;
        padding: 20px 24px;
        border-radius: 10px;
        margin: 15px 0;
    }

    .status-badge-safe h2,
    .status-badge-safe p {
        color: #ffffff !important;
    }

    /* Failure risk result badge */
    .status-badge-danger {
        background: linear-gradient(90deg, #7f1d1d, #991b1b);
        border-left: 6px solid #f87171;
        color: #ffffff !important;
        padding: 20px 24px;
        border-radius: 10px;
        margin: 15px 0;
    }

    .status-badge-danger h2,
    .status-badge-danger p {
        color: #ffffff !important;
    }

    /* Educational disclaimer */
    .disclaimer-box {
        background: #332416;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 15px 18px;
        color: #fde68a !important;
        font-size: 14px;
        line-height: 1.7;
        margin: 15px 0 25px;
    }

    .disclaimer-box strong {
        color: #fef3c7 !important;
    }

    /* Input fields */
    .stApp input,
    .stApp textarea {
        color: #f8fafc !important;
        background-color: #172a40 !important;
        border: 1px solid #52708f !important;
        border-radius: 6px;
    }

    /* Select boxes */
    .stApp [data-baseweb="select"] > div {
        background-color: #172a40 !important;
        border-color: #52708f !important;
    }

    .stApp [data-baseweb="select"] *,
    .stApp [data-baseweb="popover"] * {
        color: #f8fafc !important;
    }

    /* Sliders and slider labels */
    .stApp [data-testid="stSlider"] {
        color: #f8fafc !important;
    }

    /* Buttons */
    .stApp button[kind="primary"],
    .stApp button[kind="secondary"],
    .stApp [data-testid="stFormSubmitButton"] button {
        font-weight: 700;
        border-radius: 8px;
    }

    /* Data tables */
    .stApp [data-testid="stDataFrame"],
    .stApp [data-testid="stTable"] {
        border: 1px solid #355477;
        border-radius: 8px;
    }

    /* Tabs */
    .stApp button[data-baseweb="tab"] {
        color: #e2e8f0 !important;
        font-weight: 600;
    }

    /* Links */
    .stApp a {
        color: #7dd3fc !important;
    }

    /* Code blocks */
    .stApp code {
        color: #fde68a !important;
        background-color: #172a40;
    }

    /* Better readability on smaller screens */
    @media (max-width: 768px) {
        .rail-header {
            padding: 16px;
        }

        .rail-header h1 {
            font-size: 22px;
        }

        .metric-card {
            padding: 14px;
        }

        .metric-card .num {
            font-size: 23px;
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# File & Model Loader (Cached for Performance)
# -----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = APP_DIR

DATA_PATH = os.path.join(PROJECT_ROOT, "railway_track_inspection.csv")

MODELS_DIR = PROJECT_ROOT

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_resource
def load_models_and_preprocessor():
    artifacts = {}
    try:
        prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        if os.path.exists(prep_path):
            artifacts["preprocessor"] = joblib.load(prep_path)

        for name in ["best_model", "random_forest", "decision_tree", "logistic_regression"]:
            m_path = os.path.join(MODELS_DIR, f"{name}.joblib")
            if os.path.exists(m_path):
                artifacts[name] = joblib.load(m_path)

        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                artifacts["metadata"] = json.load(f)
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
    return artifacts

df = load_data()
artifacts = load_models_and_preprocessor()

# -----------------------------------------------------------------------------
# Sidebar Navigation & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚆 RailGuard AI")
    st.markdown("**Railway Track Failure Risk Prediction**")
    st.caption("College Mini-Project Demonstration")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation Menu:",
        [
            "🏠 Overview & Dashboard",
            "🔍 Track Risk Predictor",
            "📊 Dataset Explorer",
            "📈 Visual Analytics (EDA)",
            "🤖 Model Evaluation & Metrics",
            "⚠️ Safety & Limitations"
        ]
    )

    st.markdown("---")
    st.markdown("### 🛠️ Active Model Selection")
    model_options = {
        "Random Forest (Champion - Best Recall)": "best_model",
        "Decision Tree (Interpretable)": "decision_tree",
        "Logistic Regression (Baseline Linear)": "logistic_regression"
    }
    selected_model_label = st.selectbox("Inference Algorithm:", list(model_options.keys()))
    selected_model_key = model_options[selected_model_label]

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 11px; color: #a0aec0; text-align: center;">
            RailGuard AI v1.0<br>
            Powered by Scikit-Learn & Streamlit<br>
            <i>Educational Academic Prototype</i>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Common Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="rail-header">
        <h1>🚆 RailGuard AI — Railway Track Failure Risk System</h1>
        <p>Machine Learning prototype for non-destructive railway inspection analysis, defect classification, and predictive maintenance risk evaluation.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ IMPORTANT EDUCATIONAL DISCLAIMER:</strong>
        This system is built using a domain-grounded <strong>synthetic educational dataset</strong> for academic demonstration. 
        It is <strong>not certified</strong> by railway regulatory bodies and must never be used to make real-world railway dispatch, speed limit, or operational safety decisions.
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Page 1: Overview & Dashboard
# -----------------------------------------------------------------------------
if nav_choice == "🏠 Overview & Dashboard":
    st.subheader("📌 System Overview & Key Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Total Track Segments</div>
                <div class="num">{len(df) if df is not None else 1800:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        normal_cnt = (df["track_failure_risk"] == 0).sum() if df is not None else 1333
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Normal Condition (Safe)</div>
                <div class="num" style="color: #48bb78;">{normal_cnt:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        fail_cnt = (df["track_failure_risk"] == 1).sum() if df is not None else 467
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Potential Failure (Alert)</div>
                <div class="num" style="color: #f56565;">{fail_cnt:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        best_acc = "91.7%"
        best_rec = "95.7%"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Best Model Recall (Fail)</div>
                <div class="num" style="color: #ecc94b;">{best_rec}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1.2, 1])
    with left_col:
        st.markdown("### 🎯 Project Motivation & Purpose")
        st.write(
            """
            Railway track failures, such as internal transverse fissures, severe rail head surface cracks, and track 
            geometry buckling, represent some of the most critical safety hazards in the transportation sector.
            
            **RailGuard AI** is designed to demonstrate how machine learning classifiers can ingest multi-sensor inspection telemetry to:
            1. **Classify Track Integrity:** Accurately categorize inspected track segments into **Normal** or **Potential Failure**.
            2. **Prioritize Maintenance:** Surface high-risk segments to railway engineering crews before fractures manifest.
            3. **Optimize Inspection Cycles:** Flag tracks where high vibration, deflection, and cumulative tonnage intersect.
            """
        )

        st.markdown("### ⚙️ Telemetry & Physical Indicators")
        st.markdown(
            """
            - **Ultrasonic Flaw Detector (dB):** Internal acoustic reflections identifying hidden micro-fissures.
            - **Surface Crack Optical Sensors (mm):** Rolling Contact Fatigue (RCF) crack propagation measurement.
            - **Track Deflection Gauges (mm):** Dynamic vertical depression testing subgrade/ballast firmness.
            - **In-Sleeper Accelerometers ($g$):** Track vibration RMS measuring geometry deterioration and fastener looseness.
            - **Thermal Rail Probes (°C):** Detection of rail buckling risks in heat or brittleness in sub-zero frost.
            """
        )

    with right_col:
        st.markdown("### 📊 Track Condition Distribution")
        fig_dist = os.path.join(FIGURES_DIR, "target_distribution.png")
        if os.path.exists(fig_dist):
            st.image(fig_dist, caption="Dataset Class Balance (1,800 Segments)")
        else:
            st.info("Distribution figure will display here once generated.")

# -----------------------------------------------------------------------------
# Page 2: Track Risk Predictor
# -----------------------------------------------------------------------------
elif nav_choice == "🔍 Track Risk Predictor":
    st.subheader("🔍 Real-Time Railway Track Risk Prediction")
    st.write("Enter track inspection measurements or select a pre-configured scenario to evaluate the failure risk.")

    # Preset Quick Loaders
    st.markdown("#### ⚡ Quick Demonstration Presets")
    preset_cols = st.columns(4)
    with preset_cols[0]:
        load_healthy = st.button("🟢 Safe Track Preset")
    with preset_cols[1]:
        load_thermal = st.button("🟠 Thermal Buckle Preset")
    with preset_cols[2]:
        load_ut_flaw = st.button("🔴 Internal Flaw Preset")
    with preset_cols[3]:
        load_deflect = st.button("🟣 Subgrade Deflection Preset")

    # Set default values based on presets
    def_corridor = "Northern-Corridor"
    def_temp = 24.0
    def_vib = 0.45
    def_ut = 15.0
    def_deflect = 1.8
    def_crack = 1.5
    def_tonnage = 55.0
    def_maint = 45

    if load_healthy:
        def_temp, def_vib, def_ut, def_deflect, def_crack, def_tonnage, def_maint = 22.0, 0.32, 12.0, 1.5, 0.5, 40.0, 30
        st.toast("Loaded 'Safe Track Preset' values.")
    elif load_thermal:
        def_temp, def_vib, def_ut, def_deflect, def_crack, def_tonnage, def_maint = 52.0, 1.35, 26.0, 3.9, 8.5, 195.0, 160
        st.toast("Loaded 'Thermal Buckle Preset' values.")
    elif load_ut_flaw:
        def_temp, def_vib, def_ut, def_deflect, def_crack, def_tonnage, def_maint = 28.0, 1.10, 56.0, 3.1, 18.0, 220.0, 210
        st.toast("Loaded 'Internal Flaw Preset' values.")
    elif load_deflect:
        def_temp, def_vib, def_ut, def_deflect, def_crack, def_tonnage, def_maint = 18.0, 1.95, 34.0, 5.8, 12.0, 260.0, 290
        st.toast("Loaded 'Subgrade Deflection Preset' values.")

    st.markdown("---")
    with st.form("prediction_form"):
        st.markdown("### 📝 Track Inspection Input Parameters")

        c1, c2 = st.columns(2)
        with c1:
            corridor = st.selectbox(
                "Railway Corridor / Subdivision:",
                ["Northern-Corridor", "Central-Freight-Trunk", "Western-Coastal-Line", "Eastern-Mineral-Belt", "Southern-Express-Route"],
                index=0
            )
            rail_temp = st.slider("Rail Temperature (°C)", -10.0, 60.0, float(def_temp), 0.5, help="Normal range: 10-38°C. Critical: <0°C (brittleness) or >45°C (thermal buckle)")
            vibration = st.slider("Track Vibration RMS (g)", 0.10, 2.50, float(def_vib), 0.05, help="Normal: 0.15-0.70g. Critical: >1.10g (corrugation / loose fasteners)")
            ultrasonic = st.slider("Ultrasonic Flaw Echo Amplitude (dB)", 5.0, 65.0, float(def_ut), 0.5, help="Normal: 8-22 dB. Critical: >32 dB (internal fissures)")

        with c2:
            deflection = st.slider("Vertical Dynamic Track Deflection (mm)", 0.8, 8.5, float(def_deflect), 0.1, help="Normal: 1.2-2.8 mm. Critical: >3.8 mm (voided ballast)")
            crack_length = st.slider("Surface Micro-Crack Length (mm)", 0.0, 35.0, float(def_crack), 0.2, help="Normal: 0-4 mm. Critical: >10 mm (Rolling Contact Fatigue)")
            tonnage = st.slider("Cumulative Track Tonnage (MGT)", 10.0, 320.0, float(def_tonnage), 5.0, help="Million Gross Tons carried by rail steel")
            maintenance_days = st.slider("Days Since Last Track Maintenance", 7, 365, int(def_maint), 5, help="Days elapsed since tamping / grinding")

        submit_btn = st.form_submit_button("⚡ Assess Track Condition & Calculate Risk", use_container_width=True)

    if submit_btn or load_healthy or load_thermal or load_ut_flaw or load_deflect:
        if "preprocessor" not in artifacts or selected_model_key not in artifacts:
            st.error("Model artifacts not loaded properly. Please verify model training.")
        else:
            # Build input DataFrame
            input_dict = {
                "corridor_name": [corridor],
                "rail_temperature_c": [rail_temp],
                "vibration_rms_g": [vibration],
                "ultrasonic_flaw_signal_db": [ultrasonic],
                "vertical_deflection_mm": [deflection],
                "surface_crack_length_mm": [crack_length],
                "cumulative_tonnage_mgt": [tonnage],
                "days_since_last_maintenance": [maintenance_days]
            }
            input_df = pd.DataFrame(input_dict)

            # Preprocess and Predict
            prep = artifacts["preprocessor"]
            model = artifacts[selected_model_key]

            X_trans = prep.transform(input_df)
            pred_class = int(model.predict(X_trans)[0])
            pred_proba = float(model.predict_proba(X_trans)[0][1]) if hasattr(model, "predict_proba") else (1.0 if pred_class == 1 else 0.0)

            st.markdown("### 📋 Inspection Assessment Result")
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                if pred_class == 0:
                    st.markdown(
                        f"""
                        <div class="status-badge-safe">
                            <h2 style="margin: 0; color: #ffffff;">🟢 NORMAL TRACK CONDITION</h2>
                            <p style="margin: 8px 0 0 0; font-size: 15px;">The track segment is operating within safe physical parameters.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="status-badge-danger">
                            <h2 style="margin: 0; color: #ffffff;">🔴 POTENTIAL FAILURE RISK</h2>
                            <p style="margin: 8px 0 0 0; font-size: 15px;">Elevated degradation indicators detected! Immediate inspection recommended.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(f"**Selected Model:** `{selected_model_label}`")
                st.markdown(f"**Failure Risk Estimate:** `{pred_proba * 100:.1f}%` *(Model probability estimate)*")
                st.progress(pred_proba)

            with res_col2:
                st.markdown("#### 🔍 Defect Indicator Summary")
                warnings = []
                if ultrasonic > 32.0:
                    warnings.append(f"⚠️ **Ultrasonic Flaw ({ultrasonic} dB)** exceeds critical threshold (32 dB). Subsurface fissure alert.")
                if crack_length > 10.0:
                    warnings.append(f"⚠️ **Surface Crack ({crack_length} mm)** exceeds safety limit (10 mm). RCF propagation risk.")
                if deflection > 3.8:
                    warnings.append(f"⚠️ **Track Deflection ({deflection} mm)** indicates ballast voids or subgrade softening.")
                if vibration > 1.1:
                    warnings.append(f"⚠️ **Track Vibration ({vibration} g)** indicates loose fasteners or geometry dip.")
                if rail_temp > 45.0:
                    warnings.append(f"⚠️ **Rail Temperature ({rail_temp} °C)** indicates high compressive buckling stress.")
                elif rail_temp < 0.0:
                    warnings.append(f"⚠️ **Sub-zero Rail Temperature ({rail_temp} °C)** indicates increased steel brittleness.")

                if warnings:
                    for w in warnings:
                        st.markdown(w)
                else:
                    st.success("✅ All sensor indicators are currently within standard operating limits.")

                st.caption("Disclaimer: Probabilities are educational model outputs and do not represent verified railway engineering certifications.")

# -----------------------------------------------------------------------------
# Page 3: Dataset Explorer
# -----------------------------------------------------------------------------
elif nav_choice == "📊 Dataset Explorer":
    st.subheader("📊 Railway Track Inspection Dataset Explorer")
    st.write("Browse, filter, and inspect the 1,800 track inspection records used to train the RailGuard AI models.")

    if df is not None:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sel_corr = st.multiselect("Filter by Corridor:", options=list(df["corridor_name"].unique()), default=list(df["corridor_name"].unique()))
        with col_f2:
            sel_cond = st.multiselect("Filter by Condition:", options=list(df["risk_category"].unique()), default=list(df["risk_category"].unique()))

        filtered_df = df[(df["corridor_name"].isin(sel_corr)) & (df["risk_category"].isin(sel_cond))]
        st.write(f"Showing **{len(filtered_df):,}** out of {len(df):,} total track segments:")
        st.dataframe(filtered_df, use_container_width=True, height=350)

        st.markdown("### 📈 Summary Statistics of Numerical Inspection Features")
        numeric_cols = [
            "rail_temperature_c", "vibration_rms_g", "ultrasonic_flaw_signal_db",
            "vertical_deflection_mm", "surface_crack_length_mm", "cumulative_tonnage_mgt",
            "days_since_last_maintenance"
        ]
        st.dataframe(df[numeric_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]], use_container_width=True)
    else:
        st.error("Dataset file not found.")

# -----------------------------------------------------------------------------
# Page 4: Visual Analytics (EDA)
# -----------------------------------------------------------------------------
elif nav_choice == "📈 Visual Analytics (EDA)":
    st.subheader("📈 Exploratory Data Analysis & Engineering Visualizations")
    st.write("Explore domain-informed relationships between track sensor metrics and failure risks.")

    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Correlation Heatmap", "📦 Boxplots by Risk", "📊 Feature Distributions", "🎯 Bivariate Defects"])

    with tab1:
        st.markdown("#### Feature Correlation Heatmap")
        st.write("Displays Pearson correlation between sensor metrics and failure risk. Notice how ultrasonic flaws and surface cracks show the strongest positive correlation.")
        f_heat = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
        if os.path.exists(f_heat):
            st.image(f_heat, use_container_width=True)

    with tab2:
        st.markdown("#### Feature Comparisons (Normal vs Potential Failure)")
        st.write("Boxplots illustrate how defective tracks systematically exhibit higher deflection, vibration, flaw signals, and longer surface cracks.")
        f_box = os.path.join(FIGURES_DIR, "boxplots_by_risk.png")
        if os.path.exists(f_box):
            st.image(f_box, use_container_width=True)

    with tab3:
        st.markdown("#### Probability Density & Distributions")
        f_dist = os.path.join(FIGURES_DIR, "feature_distributions.png")
        if os.path.exists(f_dist):
            st.image(f_dist, use_container_width=True)

    with tab4:
        st.markdown("#### Ultrasonic Flaw Signal vs Surface Crack Length")
        st.write("Bivariate scatter plot showing critical safety threshold boundary lines (32 dB flaw threshold and 10 mm crack threshold).")
        f_scat = os.path.join(FIGURES_DIR, "scatter_flaw_vs_crack.png")
        if os.path.exists(f_scat):
            st.image(f_scat, use_container_width=True)

# -----------------------------------------------------------------------------
# Page 5: Model Evaluation & Metrics
# -----------------------------------------------------------------------------
elif nav_choice == "🤖 Model Evaluation & Metrics":
    st.subheader("🤖 Comparative Model Evaluation & Confusion Matrices")
    st.write("Rigorous comparison between Logistic Regression, Decision Tree, and Random Forest on the 360-segment test set.")

    eval_json_path = os.path.join(REPORTS_DIR, "model_evaluation_metrics.json")
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        summary_rows = []
        for m_name, m_val in metrics_data.items():
            summary_rows.append({
                "Algorithm": m_name,
                "Accuracy": f"{m_val['accuracy']*100:.2f}%",
                "Recall (Failure)": f"{m_val['recall_failure']*100:.2f}%",
                "Precision (Failure)": f"{m_val['precision_failure']*100:.2f}%",
                "F1-Score (Failure)": f"{m_val['f1_failure']:.4f}",
                "ROC-AUC": f"{m_val['roc_auc']:.4f}",
                "Status": "🏆 Best Model" if m_name == "Random Forest" else "Tested"
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Why Recall for Potential Failure is the #1 Safety Priority")
    st.info(
        """
        - **False Negative (FN) - Missing a defective track:** If the model predicts 'Normal' for a cracked rail, trains continue running at high speeds, risking a catastrophic derailment!
        - **False Positive (FP) - False alarm:** If the model predicts 'Failure' for a normal track, an inspection crew performs a precautionary ultrasonic check. No one gets hurt.
        - **Conclusion:** Minimizing False Negatives (maximizing **Recall**) is the life-critical priority in railway safety engineering.
        """
    )

    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        st.markdown("#### Test Set Confusion Matrices")
        f_cm = os.path.join(FIGURES_DIR, "confusion_matrices.png")
        if os.path.exists(f_cm):
            st.image(f_cm, caption="Confusion Matrices (360 Test Segments)", use_container_width=True)

    with ev_col2:
        st.markdown("#### ROC Curves Comparison")
        f_roc = os.path.join(FIGURES_DIR, "roc_curves.png")
        if os.path.exists(f_roc):
            st.image(f_roc, caption="Receiver Operating Characteristic (ROC)", use_container_width=True)

    st.markdown("---")
    fi_col1, fi_col2 = st.columns(2)
    with fi_col1:
        st.markdown("#### Random Forest Feature Importances")
        f_fi = os.path.join(FIGURES_DIR, "feature_importances.png")
        if os.path.exists(f_fi):
            st.image(f_fi, use_container_width=True)
    with fi_col2:
        st.markdown("#### Comparative Metric Bar Chart")
        f_bar = os.path.join(FIGURES_DIR, "model_comparison_bar.png")
        if os.path.exists(f_bar):
            st.image(f_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# Page 6: Safety & Limitations
# -----------------------------------------------------------------------------
elif nav_choice == "⚠️ Safety & Limitations":
    st.subheader("⚠️ Engineering Limitations & Academic Disclaimer")
    
    st.markdown(
        """
        ### 1. Synthetic Data Disclaimer
        - **Educational Prototype:** This machine learning model has been trained on a domain-grounded synthetic dataset created for academic purposes.
        - **Not Operational Data:** Although physics-informed equations and engineering threshold distributions (e.g. ultrasonic flaw signals in dB, vertical track deflection in mm) were used, the records do not represent real-world railway track inspection or incident records.
        - **No Real Railway Dispatch Decisions:** Predictions, confidence intervals, and risk scores generated by this system must **never** be used for train dispatching, track speed restrictions, or railway operational decisions.

        ### 2. Machine Learning Model Limitations
        - **Sensor Noise & Environmental Interference:** In operational tracks, ultrasonic signals are susceptible to rail head grease, leaves, and extreme rain. Physical sensors encounter electrical noise and calibration drift.
        - **Geometry Variations:** Real track geometry vehicles measure rail cant, gauge widening, twist, and alignment across dynamic curved transitions, which require 3D multi-channel time-series modeling.
        - **Threshold Calibration:** Real-world railroad maintenance follows strict standards (such as AREMA, EN 13848, or Indian Railways Track Manual guidelines) requiring physical destructive calibration and safety factor certifications.

        ### 3. College Project Context
        This project was created as an educational machine learning demonstration showcasing:
        - Non-destructive railway inspection feature engineering.
        - End-to-end data science lifecycle (data generation, EDA, stratified split, model training).
        - Evaluation prioritizing safety-critical metrics (Recall & F1-score over raw accuracy).
        - Modern web dashboard deployment using Streamlit.
        """
    )
