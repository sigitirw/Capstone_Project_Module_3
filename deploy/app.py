import os
import io
import pickle
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------
# Config & constants
# -----------------------
st.set_page_config(page_title="Telco Churn Deployment", layout="wide", initial_sidebar_state="auto")
FEATURES = [
    'Dependents', 'tenure', 'OnlineSecurity', 'OnlineBackup',
    'InternetService', 'DeviceProtection', 'TechSupport', 'Contract',
    'PaperlessBilling', 'MonthlyCharges'
]
SAMPLE_CSV_NAME = "data_telco_customer_churn.csv"
PRIMARY_CSV_NAME = "data_telco_customer_churn.csv"
PICKLE_NAME = "telcoChurn.pkl"

# Minimal CSS for bright (not plain white) theme and nicer controls
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#f3f9ff 0%, #ffffff 100%); }
    .card { background: rgba(255,255,255,0.92); border-radius:12px; padding:14px;
            box-shadow: 0 2px 8px rgba(11, 83, 178, 0.06); }
    .stButton>button { background-color: #0b63d6; color: white; border-radius:8px; }
    .small { font-size:12px; color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def try_load_csv(paths: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Attempt to read first existing csv from paths list, return (df, path) or (None, None)."""
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df, p
            except Exception:
                # try with engine fallback
                try:
                    df = pd.read_csv(p, engine="python")
                    return df, p
                except Exception:
                    continue
    return None, None

@st.cache_data
def load_model(path: str = PICKLE_NAME):
    """Load pickle model if exists, else return None."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def ensure_required_columns(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return df[required]

def safe_predict(model, X: pd.DataFrame):
    """
    Try to call model.predict/_proba with DataFrame.
    If model expects numpy array, convert automatically.
    Returns: preds, probs (or None), classes (or None)
    """
    # Some models expect numpy arrays; pipelines usually accept DataFrame
    try:
        preds = model.predict(X)
    except Exception as e:
        # fallback to values
        try:
            preds = model.predict(X.values)
        except Exception as e2:
            raise RuntimeError(f"Model.predict failed. Details: {e} | {e2}")

    probs = None
    classes = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
        except Exception:
            try:
                probs = model.predict_proba(X.values)
            except Exception:
                probs = None
    if hasattr(model, "classes_"):
        try:
            classes = model.classes_
        except Exception:
            classes = None

    return np.array(preds), probs, classes

def map_label_to_text(label):
    """Friendly label mapping — tries common churn encodings."""
    s = str(label).lower()
    if s in ("yes", "churn", "1", "true", "y"):
        return "Churn"
    if s in ("no", "not churn", "0", "false", "n"):
        return "Not Churn"
    return str(label)

def make_sample_dataframe() -> pd.DataFrame:
    """Create tiny sample DataFrame with FEATURES to show or download."""
    sample = {
        'Dependents': ['No', 'Yes'],
        'tenure': [5, 24],
        'OnlineSecurity': ['No', 'Yes'],
        'OnlineBackup': ['No', 'Yes'],
        'InternetService': ['Fiber optic', 'DSL'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['No', 'Yes'],
        'Contract': ['Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'MonthlyCharges': [75.35, 56.95]
    }
    return pd.DataFrame(sample)

# -----------------------
# Load resources
# -----------------------
model = load_model("telcoChurn.pkl")
df_sample, sample_source = try_load_csv([PRIMARY_CSV_NAME, SAMPLE_CSV_NAME])

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.title("Telco Churn")
    st.markdown("Menu:")
    page = st.radio("", ["EDA", "Prediksi"])
    st.markdown("---")
    st.write("Files in working dir:")
    files = [f for f in os.listdir(".") if f.endswith((".csv", ".pkl"))]
    if files:
        st.write(", ".join(files))
    else:
        st.write("_No csv/pkl files found_")
    st.markdown("---")
    st.markdown("**Run instructions**")
    st.caption("1) pip install -r requirements.txt\n2) streamlit run app.py")
    st.markdown("---")
    st.caption("Notes: pickle name: `telcoChurn.pkl`. Data sample: `Churn.csv` or `data_telco_customer_churn.csv`")

# -----------------------
# EDA Page
# -----------------------
if page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Upload dataset Anda (CSV) atau gunakan sample yang ada di working directory.")

    col1, col2 = st.columns([1, 3])
    with col1:
        uploaded_file = st.file_uploader("Upload CSV untuk EDA (opsional)", type=["csv"])
        use_sample_btn = st.button("Load sample CSV from working dir")
        download_sample = st.download_button(
            "Download sample CSV (template)",
            data=make_sample_dataframe().to_csv(index=False).encode("utf-8"),
            file_name="sample_telco_churn_template.csv",
            mime="text/csv"
        )

    df = None
    source = "None"
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            source = "Uploaded file"
        except Exception as e:
            st.error(f"Gagal membaca file upload: {e}")
            st.stop()
    elif use_sample_btn:
        if df_sample is not None:
            df = df_sample.copy()
            source = f"Loaded from {sample_source}"
        else:
            st.warning("Tidak ada sample CSV di working directory.")
    else:
        # auto-load if exists
        if df_sample is not None:
            df = df_sample.copy()
            source = f"Auto-loaded from {sample_source}"

    if df is None:
        st.info("Belum ada data untuk EDA. Upload file CSV atau letakkan 'Churn.csv' di folder kerja. Gunakan sample template untuk memulai.")
        st.stop()

    st.markdown(f"**Sumber data:** {source}")
    st.subheader("Preview (first 200 rows)")
    st.dataframe(df.head(200))

    st.subheader("Ringkasan dasar")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
    with c2:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            st.write("Missing per column:")
            st.dataframe(missing.to_frame("missing_count"))
        else:
            st.write("No missing values detected (0).")
    with c3:
        st.write("Data types")
        st.dataframe(df.dtypes.astype(str).to_frame("dtype"))

    st.subheader("Unique values for model features")
    unilist = {}
    for f in FEATURES:
        if f in df.columns:
            vals = df[f].dropna().unique().tolist()
            # limit display
            if len(vals) > 100:
                vals = vals[:100] + ["..."]
            unilist[f] = vals
        else:
            unilist[f] = ["<column not present>"]
    # render as table
    max_len = max(len(v) for v in unilist.values())
    uni_rows = []
    keys = list(unilist.keys())
    for i in range(max_len):
        row = {}
        for k in keys:
            row[k] = unilist[k][i] if i < len(unilist[k]) else ""
        uni_rows.append(row)
    st.dataframe(pd.DataFrame(uni_rows))

    st.subheader("Visualisasi interaktif")
    viz_choices = st.multiselect("Pilih fitur untuk plot (default: tenure, MonthlyCharges, Contract)", options=FEATURES, default=['tenure', 'MonthlyCharges', 'Contract'])
    for col in viz_choices:
        if col not in df.columns:
            st.warning(f"{col} tidak ada di dataset — dilewati.")
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribusi: {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # if Churn exists show churn rate per category
            if 'Churn' in df.columns:
                temp = df.groupby(col)['Churn'].value_counts(normalize=False).unstack(fill_value=0)
                # churn rate
                rr = df.groupby(col)['Churn'].apply(lambda s: (s == 'Yes').mean()).reset_index(name='churn_rate')
                fig = px.bar(rr, x=col, y='churn_rate', title=f"Churn rate per {col}", text='churn_rate')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col, title=f"Counts: {col}")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("----")
    st.write("Selesai EDA. Beralih ke menu 'Prediksi' untuk inferensi single atau batch.")

# -----------------------
# Prediction Page
# -----------------------
else:
    st.header("Prediksi Customer Churn")
    st.write("Masukkan data customer untuk prediksi atau upload file CSV untuk batch prediction.")

    if model is None:
        st.warning(f"Model pickle `{PICKLE_NAME}` tidak ditemukan. Letakkan pickle di working directory agar prediksi bisa berjalan.")

    # Prepare choices for categorical inputs based on loaded sample if present
    sample_df = df_sample.copy() if df_sample is not None else None

    st.subheader("Single-record prediction")
    with st.form("single_form"):
        # Dependents
        dep_opts = sorted(sample_df['Dependents'].dropna().unique().tolist()) if sample_df is not None and 'Dependents' in sample_df.columns else ['No', 'Yes']
        dependents = st.selectbox("Dependents", options=dep_opts)

        tenure = st.number_input("tenure (bulan)", value=12, min_value=0, max_value=1000, step=1)

        os_opts = sorted(sample_df['OnlineSecurity'].dropna().unique().tolist()) if sample_df is not None and 'OnlineSecurity' in sample_df.columns else ['Yes', 'No', 'No internet service']
        online_security = st.selectbox("OnlineSecurity", options=os_opts)

        ob_opts = sorted(sample_df['OnlineBackup'].dropna().unique().tolist()) if sample_df is not None and 'OnlineBackup' in sample_df.columns else ['Yes', 'No', 'No internet service']
        online_backup = st.selectbox("OnlineBackup", options=ob_opts)

        int_opts = sorted(sample_df['InternetService'].dropna().unique().tolist()) if sample_df is not None and 'InternetService' in sample_df.columns else ['DSL', 'Fiber optic', 'No']
        internet_service = st.selectbox("InternetService", options=int_opts)

        dp_opts = sorted(sample_df['DeviceProtection'].dropna().unique().tolist()) if sample_df is not None and 'DeviceProtection' in sample_df.columns else ['Yes', 'No', 'No internet service']
        device_protection = st.selectbox("DeviceProtection", options=dp_opts)

        ts_opts = sorted(sample_df['TechSupport'].dropna().unique().tolist()) if sample_df is not None and 'TechSupport' in sample_df.columns else ['Yes', 'No', 'No internet service']
        tech_support = st.selectbox("TechSupport", options=ts_opts)

        contract_opts = sorted(sample_df['Contract'].dropna().unique().tolist()) if sample_df is not None and 'Contract' in sample_df.columns else ['Month-to-month', 'One year', 'Two year']
        contract = st.selectbox("Contract", options=contract_opts)

        paper_opts = sorted(sample_df['PaperlessBilling'].dropna().unique().tolist()) if sample_df is not None and 'PaperlessBilling' in sample_df.columns else ['Yes', 'No']
        paperless = st.selectbox("PaperlessBilling", options=paper_opts)

        monthly_charges = st.number_input("MonthlyCharges", value=70.0, min_value=0.0, max_value=100000.0, format="%.2f")

        submit_single = st.form_submit_button("Predict single")

    if submit_single:
        input_dict = {
            'Dependents': [dependents],
            'tenure': [tenure],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'InternetService': [internet_service],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'Contract': [contract],
            'PaperlessBilling': [paperless],
            'MonthlyCharges': [monthly_charges]
        }
        input_df = pd.DataFrame(input_dict)
        st.write("Input preview:")
        st.dataframe(input_df)

        if model is None:
            st.error("Tidak bisa prediksi: model tidak ditemukan.")
        else:
            # ensure model input columns exist or give warning
            try:
                # We do not force reorder here — assume pipeline handles feature selection/encoding.
                preds, probs, classes = safe_predict(model, input_df)
                pred_label = map_label_to_text(preds[0])
                st.subheader("Hasil Prediksi")
                st.metric("Predicted", pred_label)

                if probs is not None:
                    # show probability table (map to classes if available)
                    if classes is not None:
                        proba_map = {str(classes[i]): probs[0, i] for i in range(probs.shape[1])}
                        proba_df = pd.DataFrame.from_dict(proba_map, orient='index', columns=['Probability']).reset_index()
                        proba_df.columns = ['Class', 'Probability']
                        proba_df['Probability (%)'] = (proba_df['Probability'] * 100).round(2)
                        st.dataframe(proba_df[['Class', 'Probability (%)']])
                        # short explanation
                        # choose highest prob
                        top_idx = np.argmax(probs[0])
                        top_class = classes[top_idx]
                        st.write(f"Model confidence: {round(float(probs[0, top_idx]) * 100, 2)}% (class: {top_class})")
                    else:
                        st.write("Probabilities (array):")
                        st.dataframe((probs * 100).round(2))
                else:
                    st.info("Model tidak menyediakan predict_proba.")

                # short textual explanation
                st.markdown("**Explanation:** Prediksi diambil dari class dengan probabilitas terbesar. Jika ingin explanations lebih lanjut (SHAP/feature importance), tambahkan pipeline explainability pada model.")
            except KeyError as ke:
                st.error(f"Kolom input tidak cocok dengan yang model harapkan: {ke}")
            except Exception as e:
                st.error(f"Prediksi gagal: {e}")

    st.markdown("---")
    st.subheader("Batch prediction (CSV upload)")
    st.markdown("Upload CSV yang memiliki kolom minimal: " + ", ".join(FEATURES))
    batch_file = st.file_uploader("Upload CSV for batch predict", type=["csv"], key="batch")

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            batch_df = None

        if batch_df is not None:
            st.write("Preview batch data (first 10 rows)")
            st.dataframe(batch_df.head(10))
            # Check columns
            missing = [c for c in FEATURES if c not in batch_df.columns]
            if missing:
                st.error(f"File upload missing required columns: {missing}")
            elif model is None:
                st.error("Model tidak ditemukan, tidak bisa melakukan batch prediction.")
            else:
                try:
                    X = batch_df[FEATURES].copy()
                    preds, probs, classes = safe_predict(model, X)
                    batch_df["_prediction_raw"] = preds
                    # friendly mapping
                    batch_df["_prediction"] = [map_label_to_text(p) for p in preds]
                    if probs is not None:
                        if classes is not None:
                            for i, c in enumerate(classes):
                                batch_df[f"prob_{c}"] = probs[:, i]
                        else:
                            for i in range(probs.shape[1]):
                                batch_df[f"prob_{i}"] = probs[:, i]
                    st.success("Batch prediction selesai.")
                    st.dataframe(batch_df.head(100))

                    # Prepare for download: convert prob cols to percent for readability
                    out_df = batch_df.copy()
                    prob_cols = [c for c in out_df.columns if c.startswith("prob_")]
                    for pc in prob_cols:
                        out_df[pc] = (out_df[pc] * 100).round(2)
                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Gagal melakukan prediksi batch: {e}")
