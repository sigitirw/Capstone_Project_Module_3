# Filename: app.py
"""
Streamlit app untuk deployment model Telco Customer Churn
Requirements (example):
pip install streamlit pandas numpy plotly scikit-learn
Jalankan:
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
import plotly.express as px

# ------------------------------
# Config
# ------------------------------
st.set_page_config(page_title="Telco Churn - Demo", layout="wide")

# Minimal CSS to create bright (but not plain-white) theme
st.markdown(
    """
    <style>
    /* page background */
    .stApp {
      background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);
    }
    /* cards & containers */
    .card {
      background: rgba(255,255,255,0.9);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    /* accent button */
    .stButton>button {
      background: #0b6efd;
      color: white;
      border-radius: 8px;
      padding: 8px 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Helpers
# ------------------------------
FEATURES = [
    'Dependents', 'tenure', 'OnlineSecurity', 'OnlineBackup',
    'InternetService', 'DeviceProtection', 'TechSupport', 'Contract',
    'PaperlessBilling', 'MonthlyCharges'
]

DEFAULT_DATA_FILES = ['Churn.csv', 'data_telco_customer_churn.csv']

@st.cache_data
def load_csv_try(files_list):
    for f in files_list:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                return df, f
            except Exception as e:
                st.warning(f"File {f} ditemukan tapi gagal dibaca: {e}")
    return None, None

@st.cache_data
def load_model(pickle_path='telcoChurn.pkl'):
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)
        return model
    return None

def ensure_columns_for_model(df, required_cols):
    # Return df with at least required cols (order). If missing, raise descriptive error.
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Data tidak memiliki kolom yang diperlukan: {missing}")
    # reorder
    return df[required_cols]

def predict_df(model, df_input):
    # Try predict and predict_proba
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(df_input)
        preds = model.predict(df_input)
        # If binary with classes_, map probabilities
        try:
            classes = model.classes_
        except:
            classes = [0,1]
        return preds, probs, classes
    else:
        preds = model.predict(df_input)
        # no proba available
        probs = None
        return preds, probs, None

# ------------------------------
# Load resources
# ------------------------------
model = load_model('telcoChurn.pkl')
df_default, loaded_fname = load_csv_try(DEFAULT_DATA_FILES)

# ------------------------------
# Sidebar - navigation
# ------------------------------
with st.sidebar:
    st.title("Telco Churn App")
    page = st.radio("Menu", ["EDA", "Prediksi"])
    st.markdown("---")
    st.markdown("**Files in working dir:**")
    st.write(", ".join([f for f in os.listdir('.') if f.endswith('.csv') or f.endswith('.pkl')]))
    st.markdown("---")
    st.markdown("**Notes**")
    st.markdown("""
    - Model pickle: `telcoChurn.pkl`
    - Data sample: `Churn.csv` (fallback `data_telco_customer_churn.csv`)
    - Untuk menjalankan: `pip install streamlit pandas numpy plotly scikit-learn` dan `streamlit run app.py`
    """)
    st.markdown("---")

# ------------------------------
# EDA Page
# ------------------------------
if page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Upload dataset Anda atau gunakan data sample yang ada di working directory.")
    col1, col2 = st.columns([1,3])

    with col1:
        uploaded = st.file_uploader("Upload CSV untuk EDA (opsional)", type=['csv'])
        use_sample = st.button("Use sample file (if available)")

    df = None
    source_note = ""
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            source_note = "Uploaded file"
        except Exception as e:
            st.error(f"Gagal membaca file upload: {e}")
    elif use_sample:
        if df_default is not None:
            df = df_default.copy()
            source_note = f"Sample file loaded: {loaded_fname}"
        else:
            st.warning("Sample file tidak ditemukan di working directory.")
    else:
        # Auto-load sample if available silently
        if df_default is not None:
            df = df_default.copy()
            source_note = f"Auto sample loaded: {loaded_fname}"

    if df is None:
        st.info("Belum ada data. Upload file CSV atau letakkan 'Churn.csv' di folder kerja.")
        st.stop()

    st.markdown(f"**Sumber data:** {source_note}")
    st.subheader("Preview data")
    st.dataframe(df.head(200))

    # Basic summaries
    st.subheader("Ringkasan Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Jumlah baris / kolom")
        st.metric("Rows", df.shape[0], delta=None)
        st.metric("Columns", df.shape[1], delta=None)
    with c2:
        st.write("Missing values (per column)")
        missing = df.isnull().sum()
        st.dataframe(missing[missing>0].sort_values(ascending=False).to_frame("missing_count"))
    with c3:
        st.write("Tipe data")
        st.dataframe(df.dtypes.astype(str).to_frame("dtype"))

    # Show unique values for model features
    st.subheader("Unique values for model features")
    uniques = {}
    for f in FEATURES:
        if f in df.columns:
            uniques[f] = df[f].dropna().unique().tolist()[:50]  # limit to 50
        else:
            uniques[f] = "Kolom tidak ada"
    # show as dataframe
    uni_df = pd.DataFrame.from_dict({k: (v if isinstance(v, list) else [v]) for k,v in uniques.items()}, orient='index').transpose()
    st.write(uni_df.fillna(""))

    # Interactive plots
    st.subheader("Visualisasi interaktif")
    viz_cols = st.multiselect("Pilih fitur untuk plot", options=FEATURES, default=['tenure','MonthlyCharges','Contract'])
    for col in viz_cols:
        if col not in df.columns:
            st.warning(f"{col} tidak ada di data, skip plot.")
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribusi: {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # bar chart with churn rate if Churn column exists
            if 'Churn' in df.columns:
                group = df.groupby(col)['Churn'].value_counts(normalize=False).unstack(fill_value=0)
                # compute churn rate per category
                temp = df.groupby(col)['Churn'].agg(['count', lambda s: (s=='Yes').mean()])
                temp.columns = ['count', 'churn_rate']
                fig = px.bar(temp.reset_index(), x=col, y='churn_rate', title=f"Churn rate per {col}", text='churn_rate')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col, title=f"Counts: {col}")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.write("Selesai EDA. Gunakan menu 'Prediksi' untuk melakukan inferensi single/batch.")

# ------------------------------
# Prediction Page
# ------------------------------
else:
    st.header("Prediksi Churn")
    if model is None:
        st.error("Model 'telcoChurn.pkl' tidak ditemukan di working directory. Letakkan file pickle dan refresh.")
    # Prepare unique value choices from sample data if available
    if df_default is not None:
        sample = df_default.copy()
    else:
        sample = None

    st.subheader("Single customer prediction")
    with st.form("single_form"):
        # Populate choices
        def choices_for(col, default_vals=None):
            if sample is not None and col in sample.columns:
                vals = sorted(sample[col].dropna().unique().tolist())
                # Convert numeric to list-of-values for numerics? but we'll use number_input for numerics
                return vals
            else:
                return default_vals or []

        # Dependents (Yes/No)
        dep_choices = choices_for('Dependents', ['Yes','No'])
        Dependents = st.selectbox("Dependents", options=dep_choices, index=0 if dep_choices else 0)

        tenure = st.number_input("tenure (bulan)", min_value=0, max_value=240, value=12)

        online_sec_choices = choices_for('OnlineSecurity', ['Yes','No','No internet service'])
        OnlineSecurity = st.selectbox("OnlineSecurity", options=online_sec_choices, index=0 if online_sec_choices else 0)

        online_backup_choices = choices_for('OnlineBackup', ['Yes','No','No internet service'])
        OnlineBackup = st.selectbox("OnlineBackup", options=online_backup_choices, index=0 if online_backup_choices else 0)

        internet_choices = choices_for('InternetService', ['DSL','Fiber optic','No'])
        InternetService = st.selectbox("InternetService", options=internet_choices, index=0 if internet_choices else 0)

        device_choices = choices_for('DeviceProtection', ['Yes','No','No internet service'])
        DeviceProtection = st.selectbox("DeviceProtection", options=device_choices, index=0 if device_choices else 0)

        tech_choices = choices_for('TechSupport', ['Yes','No','No internet service'])
        TechSupport = st.selectbox("TechSupport", options=tech_choices, index=0 if tech_choices else 0)

        contract_choices = choices_for('Contract', ['Month-to-month','One year','Two year'])
        Contract = st.selectbox("Contract", options=contract_choices, index=0 if contract_choices else 0)

        paper_choices = choices_for('PaperlessBilling', ['Yes','No'])
        PaperlessBilling = st.selectbox("PaperlessBilling", options=paper_choices, index=0 if paper_choices else 0)

        MonthlyCharges = st.number_input("MonthlyCharges (dalam satuan sama seperti data)", min_value=0.0, max_value=10000.0, value=70.0, format="%.2f")

        submit_single = st.form_submit_button("Predict single")

    if submit_single:
        input_dict = {
            'Dependents': [Dependents],
            'tenure': [tenure],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'InternetService': [InternetService],
            'DeviceProtection': [DeviceProtection],
            'TechSupport': [TechSupport],
            'Contract': [Contract],
            'PaperlessBilling': [PaperlessBilling],
            'MonthlyCharges': [MonthlyCharges]
        }
        input_df = pd.DataFrame(input_dict)
        st.write("Input (preview):")
        st.dataframe(input_df)

        if model is None:
            st.warning("Tidak ada model untuk melakukan prediksi. Pastikan telcoChurn.pkl ada.")
        else:
            try:
                # If model expects full original columns, attempt to match
                # We assume pipeline handles encoding; otherwise user must adjust.
                preds, probs, classes = predict_df(model, input_df)
                label = preds[0]
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Hasil Prediksi (single)")
                # Map label to friendly
                try:
                    # If label is 'Yes'/'No' or 1/0
                    display_label = "Churn" if str(label).lower() in ['yes','1','true','churn'] else "Not Churn"
                except:
                    display_label = str(label)
                st.metric("Predicted label", display_label)

                if probs is not None:
                    # find index for 'Yes' if classes provided, else show both
                    prob_text = ""
                    if classes is not None:
                        # build mapping
                        proba_map = {str(c): probs[0, i] for i, c in enumerate(classes)}
                        st.write("Probabilities:")
                        proba_df = pd.DataFrame([proba_map])
                        st.dataframe((proba_df*100).round(2).T.rename(columns={0:'Probability (%)'}))
                    else:
                        st.write("Probabilities (array):")
                        st.dataframe(np.round(probs*100,2))
                else:
                    st.info("Model tidak menyediakan predict_proba.")
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")

    st.markdown("---")
    st.subheader("Batch prediction (upload CSV)")
    st.markdown("Upload CSV yang setidaknya memiliki kolom: " + ", ".join(FEATURES))
    batch_file = st.file_uploader("Upload CSV untuk batch predict", type=['csv'], key='batch')
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
        except Exception as e:
            st.error(f"Gagal membaca file CSV: {e}")
            batch_df = None

        if batch_df is not None:
            st.write("Preview batch data")
            st.dataframe(batch_df.head(10))

            # check for required columns
            missing_cols = [c for c in FEATURES if c not in batch_df.columns]
            if missing_cols:
                st.error(f"File upload missing kolom: {missing_cols}. Sesuaikan CSV dan upload ulang.")
            else:
                try:
                    input_for_model = batch_df[FEATURES].copy()
                    preds, probs, classes = predict_df(model, input_for_model)
                    batch_df['_prediction'] = preds
                    if probs is not None:
                        # attach probabilities as columns
                        if classes is not None:
                            for i, c in enumerate(classes):
                                batch_df[f'prob_{c}'] = probs[:, i]
                        else:
                            # two columns
                            for i in range(probs.shape[1]):
                                batch_df[f'prob_{i}'] = probs[:, i]
                    st.success("Prediksi selesai.")
                    st.dataframe(batch_df.head(50))

                    # provide download
                    to_download = batch_df.copy()
                    # convert probs to percent
                    prob_cols = [c for c in to_download.columns if c.startswith('prob_')]
                    if prob_cols:
                        to_download[prob_cols] = (to_download[prob_cols]*100).round(2)

                    csv_buf = to_download.to_csv(index=False).encode('utf-8')
                    st.download_button("Download predictions CSV", data=csv_buf, file_name="predictions.csv", mime="text/csv")

                except Exception as e:
                    st.error(f"Gagal melakukan prediksi batch: {e}")

    st.markdown("---")
    st.write("Catatan: Aplikasi ini mengasumsikan `telcoChurn.pkl` adalah pipeline/estimator yang menerima DataFrame dengan kolom fitur yang sesuai. Jika model Anda memerlukan preprocessing khusus, adaptasikan bagian input/preprocessing di kode ini.")
