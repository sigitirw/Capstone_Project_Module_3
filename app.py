# app.py
"""
Streamlit app (single-file) untuk deployment model klasifikasi Customer Churn.
- EDA interaktif (upload CSV atau baca telco_customer_churn.csv dari working dir)
- Predict single customer (form)
- Predict from file (batch upload + download)
- Load model from telcoChurn.pkl (fallback churn.pkl)
- Save prediction logs to predictions_log.csv
- Lightweight explainability: feature_importances_ or SHAP if available
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io
import warnings

warnings.filterwarnings("ignore")

# Try optional SHAP import (used if available)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ========== CONFIG ==========

# Visual / theme variables (bright but not white)
PAGE_BG = "#f0f9f9"    # pastel very-light
CARD_BG = "#ffffff"
ACCENT = "#0ea5a4"     # blue-tosca accent
ACCENT2 = "#ff6b6b"    # alternative accent (coral)

# Filenames (try prioritized list)
DATA_FILENAME_CANDIDATES = ["telco_customer_churn.csv", "data_telco_customer_churn.csv", "data_telco_customer_churn (1).csv"]
MODEL_FILENAME_CANDIDATES = ["telcoChurn.pkl", "churn.pkl", "telco_churn.pkl"]

# Features of interest (as requested)
FEATURES = [
    'Dependents', 'tenure', 'OnlineSecurity', 'OnlineBackup',
    'InternetService', 'DeviceProtection', 'TechSupport', 'Contract',
    'PaperlessBilling', 'MonthlyCharges'
]

# Page config
st.set_page_config(page_title="Telco Churn Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (simple)
st.markdown(
    f"""
    <style>
    .stApp {{ background: {PAGE_BG}; }}
    .card {{ background: {CARD_BG}; padding: 18px; border-radius: 12px; box-shadow: 0 6px 20px rgba(14,165,164,0.06); }}
    .title {{ font-size:26px; font-weight:700; color:#083344; }}
    .muted {{ color:#4b6b6b; }}
    .small-muted {{ color:#6b7f7f; font-size:12px; }}
    .accent-btn {{ background-color: {ACCENT}; color: white; border-radius:6px; padding:6px 10px; }}
    </style>
    """, unsafe_allow_html=True
)

# ========== HELPERS ==========

def find_existing_file(candidates):
    """Return first existing filename from list or None."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def load_model_try(paths=MODEL_FILENAME_CANDIDATES):
    """Try to load model from list of candidate filenames. Return (model, filename, error)"""
    for p in paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                return model, p, None
            except Exception as e:
                return None, p, f"File found but failed to load: {e}"
    return None, None, f"No model file found. Tried: {', '.join(paths)}"

def read_csv_from(uploaded_file, fallback_candidates=DATA_FILENAME_CANDIDATES):
    """Read CSV from uploaded_file (Streamlit uploader) or fallback file in working dir."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df, None
        except Exception as e:
            return None, f"Failed to read uploaded CSV: {e}"
    else:
        existing = find_existing_file(fallback_candidates)
        if existing:
            try:
                df = pd.read_csv(existing)
                return df, None
            except Exception as e:
                return None, f"Found {existing} but failed to read: {e}"
        else:
            return None, "No CSV uploaded and no fallback data file found."

def df_summary(df):
    """Return summary dict for the dataframe."""
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "unique": {c: int(df[c].nunique()) for c in df.columns}
    }

def prepare_single_input_dict(values: dict):
    """Return dataframe single-row containing FEATURES (order preserved). Missing features left as NaN."""
    # ensure all FEATURES present; if not provided, set np.nan
    row = {f: values.get(f, np.nan) for f in FEATURES}
    return pd.DataFrame([row])

def safe_predict(model, X: pd.DataFrame):
    """
    Safely call model.predict and model.predict_proba if available.
    Returns (preds, probs, error_msg)
    probs will be ndarray shape (n_samples, n_classes) or None
    """
    try:
        preds = model.predict(X)
    except Exception as e:
        return None, None, f"model.predict failed: {e}"

    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
        except Exception as e:
            # continue without probs
            probs = None
    return preds, probs, None

def map_readable_label(pred):
    """
    Map model output to readable label string 'Churn' or 'Not Churn'.
    Handles common encodings.
    """
    # support array-like and scalar
    def single(p):
        if isinstance(p, (int, np.integer)):
            # common: 1 -> churn, 0 -> not churn
            return "Churn" if int(p) == 1 else "Not Churn"
        if isinstance(p, str):
            p_low = p.lower()
            if p_low in ["yes", "y", "churn", "true", "1"]:
                return "Churn"
            else:
                return "Not Churn"
        if p is True:
            return "Churn"
        return "Not Churn"
    if isinstance(pred, (list, np.ndarray, pd.Series)):
        return [single(p) for p in pred]
    else:
        return single(pred)

def prob_for_churn(probs, classes=None):
    """
    Given probs array and optional classes (model.classes_), attempt to return the probability of churn for each sample.
    If classes provided and contains a 'Churn' or 'Yes' or 1 mapping, use that index.
    Otherwise, if binary assume column 1 corresponds to positive class.
    Returns 1D array of churn probabilities.
    """
    if probs is None:
        return None
    probs = np.asarray(probs)
    if probs.ndim == 1:
        # weird one-dim; return as-is
        return probs
    if classes is not None:
        # try to find index of 'Churn' class
        classes_list = [str(c).lower() for c in classes]
        for target in ['churn', 'yes', '1', 'true']:
            if target in classes_list:
                idx = classes_list.index(target)
                return probs[:, idx]
    # fallback: if two columns, return second
    if probs.shape[1] >= 2:
        return probs[:, 1]
    # fallback: return max column
    return probs.max(axis=1)

def save_prediction_log(log_row: dict, log_filename="predictions_log.csv"):
    """Append a dict row to CSV (create file if not exists)."""
    df = pd.DataFrame([log_row])
    header = not os.path.exists(log_filename)
    try:
        df.to_csv(log_filename, mode='a', header=header, index=False)
        return True, None
    except Exception as e:
        return False, str(e)

# ========== APP LAYOUT & BEHAVIOR ==========

# Load model once
model, model_path, model_err = load_model_try()

# Sidebar navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih halaman:", ["Home", "Exploratory Data Analysis (EDA)", "Predict single customer", "Predict from file"])

st.sidebar.markdown("---")
if model is not None:
    st.sidebar.success(f"Model loaded: {model_path}")
else:
    st.sidebar.error(f"Model not loaded. {model_err}")
st.sidebar.markdown("Data file for EDA (fallback): `telco_customer_churn.csv` (or upload).")
st.sidebar.markdown("---")
st.sidebar.markdown("Requirements: streamlit, pandas, scikit-learn, plotly, joblib, matplotlib, seaborn (optional), shap (optional)")

# Header
col_left, col_right = st.columns([9,1])
with col_left:
    st.markdown('<div class="title">Telco Customer Churn App</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Interactive EDA & model inference — upload dataset or use default CSV. Single-file app.</div>', unsafe_allow_html=True)
with col_right:
    # optional small logo; if you have local image, replace with st.image("logo.png")
    st.image("https://raw.githubusercontent.com/plotly/datasets/master/logo/newplotly.png", width=64)

st.markdown("---")

# ---------- HOME ----------
if page == "Home":
    st.header("Welcome")
    st.write("""
        Aplikasi ini menyediakan:
        - Exploratory Data Analysis (EDA) interaktif untuk dataset Telco Churn.
        - Prediksi single customer (form input) dengan probabilitas dan indikator confidence.
        - Prediksi batch via upload CSV dengan hasil yang dapat didownload.
        """)
    st.info("Pastikan file model pickle ada di working directory. App mencoba load: " + ", ".join(MODEL_FILENAME_CANDIDATES))

# ---------- EDA ----------
if page == "Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Upload file CSV (`telco_customer_churn.csv`) untuk EDA. Jika tidak diupload, app akan mencoba membaca fallback file dari working directory.")

    uploaded = st.file_uploader("Upload CSV for EDA", type=["csv"])
    df, err = read_csv_from(uploaded, fallback_candidates=DATA_FILENAME_CANDIDATES)
    if df is None:
        st.warning(err)
        st.stop()

    # Show preview
    st.subheader("Dataset preview")
    st.dataframe(df.head(200))

    # Summary metrics
    s = df_summary(df)
    st.subheader("Dataset summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", s["rows"])
    c2.metric("Columns", s["cols"])
    c3.metric("Missing cells", int(df.isna().sum().sum()))

    # dtypes & uniques
    st.markdown("**Tipe data tiap kolom**")
    dtypes_df = pd.DataFrame.from_dict(s["dtypes"], orient="index", columns=["dtype"])
    st.table(dtypes_df)

    st.markdown("**Unique counts (features of interest)**")
    uc = {f: s["unique"].get(f, "N/A") for f in FEATURES}
    st.json(uc)

    st.markdown("---")
    st.subheader("Feature Visualizations")
    # allow selecting which feature to inspect
    available_feats = [c for c in FEATURES if c in df.columns]
    if not available_feats:
        st.info("Tidak ada fitur interest (FEATURES) dalam dataset. Pastikan dataset memiliki kolom yang sesuai.")
    else:
        feat = st.selectbox("Pilih fitur untuk visualisasi", available_feats)
        if feat:
            if pd.api.types.is_numeric_dtype(df[feat]):
                st.markdown(f"**{feat} — Histogram + Boxplot**")
                fig = px.histogram(df, x=feat, nbins=30, marginal="box", title=f"Histogram {feat}")
                st.plotly_chart(fig, use_container_width=True)
                st.write(df[feat].describe())
            else:
                st.markdown(f"**{feat} — Count plot & unique values**")
                vc = df[feat].value_counts(dropna=False).reset_index()
                vc.columns = [feat, "count"]
                fig = px.bar(vc, x=feat, y="count", title=f"Counts of {feat}")
                st.plotly_chart(fig, use_container_width=True)
                st.table(vc)

    # Specific plots for tenure and MonthlyCharges
    st.markdown("---")
    st.subheader("Tenure & MonthlyCharges analysis")
    if 'tenure' in df.columns:
        fig = px.histogram(df, x='tenure', nbins=40, marginal="box", title="tenure distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.write(df['tenure'].describe())
    if 'MonthlyCharges' in df.columns:
        fig = px.histogram(df, x='MonthlyCharges', nbins=40, marginal="box", title="MonthlyCharges distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.write(df['MonthlyCharges'].describe())

    # InternetService and Contract bar charts
    st.markdown("---")
    st.subheader("InternetService & Contract")
    if 'InternetService' in df.columns:
        fig = px.bar(df['InternetService'].value_counts().reset_index().rename(columns={'index':'InternetService','InternetService':'count'}), x='InternetService', y='count', title='InternetService counts')
        st.plotly_chart(fig, use_container_width=True)
    if 'Contract' in df.columns:
        fig = px.bar(df['Contract'].value_counts().reset_index().rename(columns={'index':'Contract','Contract':'count'}), x='Contract', y='count', title='Contract counts')
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap for numeric features
    st.markdown("---")
    st.subheader("Correlation heatmap (numeric features)")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        fig = px.imshow(numeric.corr(), text_auto=True, aspect="auto", title="Numeric correlation")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak cukup numeric columns untuk correlation heatmap.")

    # Churn vs Contract
    if 'Churn' in df.columns:
        st.markdown("---")
        st.subheader("Churn distribution & Churn vs Contract")
        churn_counts = df['Churn'].value_counts().rename_axis('Churn').reset_index(name='count')
        fig = px.pie(churn_counts, names='Churn', values='count', title='Churn ratio')
        st.plotly_chart(fig, use_container_width=True)

        if 'Contract' in df.columns:
            ct = pd.crosstab(df['Contract'], df['Churn']).reset_index()
            # melt to long for plotting
            ct_long = ct.melt(id_vars='Contract', var_name='Churn', value_name='count')
            fig = px.bar(ct_long, x='Contract', y='count', color='Churn', barmode='group', title='Contract vs Churn')
            st.plotly_chart(fig, use_container_width=True)

    # Filters
    st.markdown("---")
    st.subheader("Filter dataset (dynamic)")
    with st.expander("Open filters"):
        filters = {}
        if 'Contract' in df.columns:
            contracts = st.multiselect("Contract", options=sorted(df['Contract'].dropna().unique().tolist()), default=sorted(df['Contract'].dropna().unique().tolist()))
            if contracts:
                filters['Contract'] = contracts
        if 'tenure' in df.columns:
            tmin = int(df['tenure'].min())
            tmax = int(df['tenure'].max())
            tenure_range = st.slider("tenure range", min_value=tmin, max_value=tmax, value=(tmin, tmax))
            filters['tenure'] = tenure_range

        # apply filters
        dff = df.copy()
        if 'Contract' in filters:
            dff = dff[dff['Contract'].isin(filters['Contract'])]
        if 'tenure' in filters:
            dff = dff[(dff['tenure'] >= filters['tenure'][0]) & (dff['tenure'] <= filters['tenure'][1])]
        st.write(f"Filtered rows: {dff.shape[0]}")
        st.dataframe(dff.head(200))

# ---------- PREDICT SINGLE ----------
if page == "Predict single customer":
    st.header("Predict single customer")
    st.markdown("Masukkan input customer lalu tekan **Predict**. Model diambil dari working directory (telcoChurn.pkl atau churn.pkl).")

    if model is None:
        st.error("Model tidak tersedia. Letakkan salah satu dari: " + ", ".join(MODEL_FILENAME_CANDIDATES))
        st.stop()

    with st.form("single_predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            Dependents = st.selectbox("Dependents", options=["Yes", "No"], index=1, help="Apakah customer punya dependents?")
            tenure = st.number_input("tenure (months)", min_value=0, max_value=500, value=12, help="Jumlah bulan customer bertahan")
            OnlineSecurity = st.selectbox("OnlineSecurity", options=["Yes", "No"], index=1, help="Online security subscription?")
            OnlineBackup = st.selectbox("OnlineBackup", options=["Yes", "No"], index=1, help="Online backup subscription?")
            InternetService = st.selectbox("InternetService", options=["DSL", "Fiber optic", "No"], index=0, help="Jenis internet service")
        with col2:
            DeviceProtection = st.selectbox("DeviceProtection", options=["Yes", "No"], index=1, help="Device protection subscription?")
            TechSupport = st.selectbox("TechSupport", options=["Yes", "No"], index=1, help="Tech support subscription?")
            Contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"], index=0, help="Jenis kontrak")
            PaperlessBilling = st.selectbox("PaperlessBilling", options=["Yes", "No"], index=0, help="Paperless billing enabled?")
            MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=10000.0, value=70.0, format="%.2f", help="Biaya bulanan (angka desimal)")

        submit = st.form_submit_button("Predict")

    if submit:
        # prepare input dict and df
        input_values = {
            'Dependents': Dependents,
            'tenure': int(tenure),
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'InternetService': InternetService,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'MonthlyCharges': float(MonthlyCharges)
        }
        X_single = prepare_single_input_dict(input_values)

        # Try to align columns to what model expects if model has 'feature_names_in_'
        # or is a pipeline with a ColumnTransformer — we send full DataFrame and rely on pipeline.
        preds, probs, perr = safe_predict(model, X_single)
        if perr:
            st.error(f"Error saat prediksi: {perr}")
        else:
            readable = map_readable_label(preds)
            # determine churn probability
            churn_prob_arr = prob_for_churn(probs, getattr(model, 'classes_', None))
            churn_prob = float(churn_prob_arr[0]) if churn_prob_arr is not None else None

            st.markdown("### Result")
            st.write(f"**Predicted label:** **{readable[0]}**")
            if churn_prob is not None:
                st.write(f"**Probability of churn:** {churn_prob*100:.2f}%")
                # confidence badge
                if churn_prob < 0.6:
                    st.warning("Low confidence (probability < 60%)")
                # gauge
                fig = go.Figure(go.Indicator(mode="gauge+number", value=churn_prob*100,
                                             title={'text': "Churn probability (%)"},
                                             gauge={'axis': {'range': [0,100]}}))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model tidak mengembalikan probabilities.")

            # Save log
            log_row = input_values.copy()
            log_row.update({
                'prediction': readable[0],
                'prob_churn': churn_prob if churn_prob is not None else '',
                'model_file': model_path,
                'timestamp': datetime.utcnow().isoformat()
            })
            ok, log_err = save_prediction_log(log_row)
            if not ok:
                st.info(f"Gagal menyimpan log: {log_err}")
            else:
                st.success("Prediction logged to predictions_log.csv")

            # Lightweight explainability: feature_importances_ if present
            st.markdown("---")
            st.subheader("Explainability (lightweight)")
            if hasattr(model, "feature_importances_"):
                try:
                    importances = model.feature_importances_
                    # if model has feature_names_in_
                    if hasattr(model, "feature_names_in_"):
                        names = list(model.feature_names_in_)
                    else:
                        # fallback to FEATURES list (best effort)
                        names = FEATURES
                    df_imp = pd.DataFrame({"feature": names[:len(importances)], "importance": importances}).sort_values("importance", ascending=False)
                    fig = px.bar(df_imp, x='importance', y='feature', orientation='h', title="Feature importances")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info("Gagal menampilkan feature_importances_: " + str(e))
            elif SHAP_AVAILABLE:
                st.info("SHAP tersedia: melakukan explainability (may take time).")
                try:
                    explainer = shap.Explainer(model.predict, X_single)
                    shap_values = explainer(X_single)
                    # Use shap.plots.force via matplotlib? We'll render a simple bar of mean absolute shap
                    shap_df = pd.DataFrame({'feature': X_single.columns, 'mean_abs_shap': np.abs(shap_values.values[0]).tolist()})
                    shap_df = shap_df.sort_values('mean_abs_shap', ascending=True)
                    fig = px.bar(shap_df, x='mean_abs_shap', y='feature', orientation='h', title="|SHAP value| per feature")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info("SHAP explainability gagal: " + str(e))
            else:
                st.info("No feature_importances_ or SHAP available for explainability.")

# ---------- PREDICT FROM FILE (BATCH) ----------
if page == "Predict from file":
    st.header("Predict from file (batch)")
    st.markdown("Upload CSV yang berisi kolom fitur yang sama seperti saat training. App akan menambahkan kolom `prediction` dan `prob_churn`.")

    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    if uploaded_file is None:
        st.info("Silakan upload CSV untuk melakukan prediksi batch.")
        st.stop()
    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    st.write(f"Uploaded file with {batch_df.shape[0]} rows")
    st.dataframe(batch_df.head(10))

    if model is None:
        st.error("Model tidak tersedia. Prediksi batch tidak dapat dijalankan.")
        st.stop()

    if st.button("Run batch prediction"):
        # try predict
        preds, probs, perr = safe_predict(model, batch_df)
        if perr:
            st.error(f"Error saat prediksi: {perr}")
        else:
            readable_preds = map_readable_label(preds)
            batch_df['prediction'] = readable_preds
            churn_probs = prob_for_churn(probs, getattr(model, 'classes_', None))
            if churn_probs is not None:
                batch_df['prob_churn'] = churn_probs
            else:
                batch_df['prob_churn'] = np.nan

            st.success("Batch prediction complete")
            st.dataframe(batch_df.head(50))

            # Offer download
            csv_bytes = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", csv_bytes, file_name="predictions_with_scores.csv", mime="text/csv")

            # Save log (append)
            try:
                batch_log = batch_df.copy()
                batch_log['model_file'] = model_path
                batch_log['timestamp'] = datetime.utcnow().isoformat()
                # append to predictions_log.csv (without header if exists)
                header = not os.path.exists("predictions_log.csv")
                batch_log.to_csv("predictions_log.csv", mode='a', header=header, index=False)
                st.success("Batch predictions appended to predictions_log.csv")
            except Exception as e:
                st.info("Gagal menyimpan batch log: " + str(e))

# End of file
