i# Streamlit single-file app for Telco Customer Churn
# Filename: app.py
# Requirements / Run instructions (also shown inside the app):
# 1. Install: pip install -r requirements.txt
# 2. Run: streamlit run app.py
# requirements.txt (short):
# streamlit
# pandas
# numpy
# joblib
# scikit-learn
# plotly
# matplotlib
# NOTE: This file assumes two files may exist in the current working dir:
# - data_telco_customer_churn.csv (default dataset for EDA)
# - telcoChurn.pkl (pickled scikit-learn pipeline: preprocessing + estimator)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.exceptions import NotFittedError

# --------------------------
# App constants & styling
# --------------------------
APP_TITLE = "Telco Customer Churn Explorer"
APP_DESCRIPTION = "A lightweight Streamlit app for EDA and deploying a churn classification model."
PASTEL_BG = "#f2fbfa"
ACCENT = "#0ea5a4"  # teal accent
DEFAULT_DATAFILE = "data_telco_customer_churn.csv"
MODEL_FILE = "telcoChurn.pkl"
# LOG_FILE changed as requested — WARNING: this will overwrite/append to your dataset file if same name
LOG_FILE = "data_telco_customer_churn.csv"

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

# Custom CSS for light pastel background and white cards
st.markdown(f"""
<style>
body, .reportview-container, .main {{background-color: {PASTEL_BG};}}
.css-18e3th9 {{padding-top: 1rem}} /* adjust top padding */
.header-card {{background: white; border-radius:12px; padding: 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.08);}}
.side-card {{background: white; border-radius:12px; padding: 12px; box-shadow: 0 3px 10px rgba(0,0,0,0.06);}}
.small-muted {{color: #666; font-size:12px}}
.btn-accent {{background-color: {ACCENT}; color: white}}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Utility functions
# --------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    """Try joblib.load then pickle.load. Returns model or raises exception."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    try:
        return joblib.load(path)
    except Exception:
        with open(path, 'rb') as f:
            return pickle.load(f)


def safe_model_load(path):
    try:
        model = load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)


def ensure_columns_for_model(X, required_cols):
    missing = [c for c in required_cols if c not in X.columns]
    return missing


def append_prediction_log(record: dict, logfile=LOG_FILE):
    df = pd.DataFrame([record])
    if os.path.exists(logfile):
        # append without header to avoid rewriting header of CSV
        df.to_csv(logfile, mode='a', header=False, index=False)
    else:
        # create file (this will create the file with header)
        df.to_csv(logfile, index=False)


# --------------------------
# Page: Header and Sidebar
# --------------------------
with st.container():
    st.markdown('<div class="header-card">', unsafe_allow_html=True)
    cols = st.columns([0.12, 0.88])
    with cols[0]:
        # small placeholder logo
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/88/OOjs_UI_icon_edit-ltr-progressive.svg", width=60)
    with cols[1]:
        st.markdown(f"<h1 style='margin:0;color:#083344'>{APP_TITLE}</h1>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>{APP_DESCRIPTION}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<div class="side-card">', unsafe_allow_html=True)
menu = st.sidebar.radio("Menu", ["Home", "EDA", "Predict single customer", "Predict from file"])  # left by design
st.sidebar.markdown('---')
# Model status in sidebar (load by explicit path 'telcoChurn.pkl')
model, model_err = safe_model_load('telcoChurn.pkl')
if model is not None:
    st.sidebar.success("Model loaded: telcoChurn.pkl")
else:
    st.sidebar.error("Model not loaded")
    st.sidebar.info("Expected model filename: telcoChurn.pkl (scikit-learn pipeline). Place in working directory.")
    if model_err:
        st.sidebar.caption(model_err)
st.sidebar.markdown('---')
st.sidebar.markdown("**Run instructions**")
st.sidebar.markdown("`pip install -r requirements.txt`\n`streamlit run app.py`")
st.sidebar.markdown('<div class="small-muted">Requirements: streamlit, pandas, numpy, joblib, scikit-learn, plotly</div>', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# Helper: load default or uploaded data
# --------------------------
@st.cache_data
def get_default_dataframe(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df, "uploaded"
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            return None, None
    else:
        # try working dir
        if os.path.exists(DEFAULT_DATAFILE):
            try:
                df = pd.read_csv(DEFAULT_DATAFILE)
                return df, "default"
            except Exception as e:
                st.warning(f"Could not read default file {DEFAULT_DATAFILE}: {e}")
                return None, None
        else:
            return None, None


# --------------------------
# Home page
# --------------------------
if menu == "Home":
    st.header("Welcome")
    st.markdown("This app provides interactive EDA and a lightweight prediction UI for telco churn models.")
    st.markdown("#### Quick checklist")
    st.markdown("- Put `telcoChurn.pkl` (scikit-learn pipeline) in working dir to enable predictions.\n- Optional: place `data_telco_customer_churn.csv` in working dir for default EDA.")
    st.info("If model is missing, prediction sections will be disabled and show instructions.")
    st.markdown('---')
    st.subheader("About the data fields used in UI")
    st.markdown("`Dependents, tenure, OnlineSecurity, OnlineBackup, InternetService, DeviceProtection, TechSupport, Contract, PaperlessBilling, MonthlyCharges`")

# --------------------------
# EDA page
# --------------------------
elif menu == "EDA":
    st.header("Exploratory Data Analysis (interactive)")
    uploaded = st.file_uploader("Upload CSV for EDA (optional)", type=['csv'])
    df, source = get_default_dataframe(uploaded)
    if df is None:
        st.warning(f"No data available. Provide `{DEFAULT_DATAFILE}` in working dir or upload a CSV.")
    else:
        st.success(f"Using dataset from: {source}")
        st.subheader("Preview & basic info")
        cols1, cols2 = st.columns(2)
        with cols1:
            st.write(df.head())
            st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
        with cols2:
            st.write(df.dtypes)
            miss = df.isna().sum()
            st.write(pd.DataFrame({'missing': miss[miss>0]}))

        st.markdown('---')
        # Focus features
        focus = ['Dependents', 'tenure', 'OnlineSecurity', 'OnlineBackup', 'InternetService', 'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling', 'MonthlyCharges']
        st.subheader("Focused feature overview")
        for f in focus:
            if f in df.columns:
                if df[f].dtype == 'object' or pd.api.types.is_categorical_dtype(df[f]):
                    vc = df[f].value_counts(dropna=False)
                    st.markdown(f"**{f}** — unique values: {df[f].nunique()}")
                    st.write(vc.reset_index().rename(columns={'index':f, f:f+'_count'}))
                else:
                    st.markdown(f"**{f}** — numeric stats")
                    st.write(df[f].describe())
            else:
                st.markdown(f"**{f}** — *not found in dataset*", unsafe_allow_html=True)

        st.markdown('---')
        # Interactive filters
        st.subheader("Interactive visualizations")
        filter_col1, filter_col2 = st.columns([1,2])
        with filter_col1:
            if 'Contract' in df.columns:
                contract_opts = df['Contract'].dropna().unique().tolist()
            else:
                contract_opts = []
            chosen_contracts = st.multiselect("Filter: Contract", options=contract_opts, default=contract_opts)
            tenure_min = int(df['tenure'].min()) if 'tenure' in df.columns else 0
            tenure_max = int(df['tenure'].max()) if 'tenure' in df.columns else 72
            chosen_tenure = st.slider("Filter: tenure range", min_value=tenure_min, max_value=tenure_max, value=(tenure_min, tenure_max), help="Filter displayed rows by tenure")
        # apply filters
        df_viz = df.copy()
        if chosen_contracts:
            if 'Contract' in df_viz.columns:
                df_viz = df_viz[df_viz['Contract'].isin(chosen_contracts)]
        if 'tenure' in df_viz.columns:
            df_viz = df_viz[(df_viz['tenure']>=chosen_tenure[0]) & (df_viz['tenure']<=chosen_tenure[1])]

        # churn distribution
        if 'Churn' in df_viz.columns:
            st.subheader('Churn distribution')
            churn_counts = df_viz['Churn'].value_counts(dropna=False).reset_index()
            churn_counts.columns = ['Churn','count']
            fig = px.pie(churn_counts, names='Churn', values='count', title='Churn share')
            st.plotly_chart(fig, use_container_width=True)

            if 'Contract' in df_viz.columns:
                st.subheader('Churn vs Contract')
                fig2 = px.histogram(df_viz, x='Contract', color='Churn', barmode='group', title='Churn by Contract')
                st.plotly_chart(fig2, use_container_width=True)

        # categorical bar charts for selected features
        cat_features = [f for f in focus if f in df.columns and (df[f].dtype=='object' or pd.api.types.is_categorical_dtype(df[f]))]
        for f in cat_features:
            fig = px.histogram(df_viz, x=f, title=f"Count by {f}")
            st.plotly_chart(fig, use_container_width=True)

        # numeric distributions
        num_features = [f for f in ['tenure', 'MonthlyCharges'] if f in df.columns]
        for f in num_features:
            st.subheader(f"Distribution: {f}")
            fig_hist = px.histogram(df_viz, x=f, nbins=40, marginal='box', title=f"Histogram + Boxplot for {f}")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.write(df_viz[f].describe())

        # correlation heatmap for numeric columns
        st.subheader('Correlation heatmap (numeric)')
        numcols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
        if len(numcols) >= 2:
            corr = df_viz[numcols].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation matrix')
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info('Not enough numeric columns for correlation matrix')

# --------------------------
# Predict single customer
# --------------------------
elif menu == "Predict single customer":
    st.header("Predict single customer")
    if model is None:
        st.error("Prediction disabled because model could not be loaded. Place telcoChurn.pkl in working directory.")
    else:
        st.markdown("Use the form to input a single customer's data and get a churn probability.")
        with st.form('single_predict_form'):
            col1, col2 = st.columns(2)
            with col1:
                Dependents = st.selectbox('Dependents', options=['No','Yes'], index=0, help='Does the customer have dependents?')
                tenure = st.number_input('tenure (months)', min_value=0, max_value=1000, value=12, step=1, help='Length of stay in months')
                OnlineSecurity = st.selectbox('OnlineSecurity', options=['No','Yes'], index=0)
                OnlineBackup = st.selectbox('OnlineBackup', options=['No','Yes'], index=0)
                DeviceProtection = st.selectbox('DeviceProtection', options=['No','Yes'], index=0)
            with col2:
                TechSupport = st.selectbox('TechSupport', options=['No','Yes'], index=0)
                PaperlessBilling = st.selectbox('PaperlessBilling', options=['No','Yes'], index=0)
                InternetService = st.selectbox('InternetService', options=['DSL','Fiber optic','No'], index=0)
                Contract = st.selectbox('Contract', options=['Month-to-month','One year','Two year'], index=0)
                MonthlyCharges = st.number_input('MonthlyCharges', min_value=0.0, value=70.0, step=0.1)

            submitted = st.form_submit_button('Predict', help='Run model prediction')
            if submitted:
                # Build dataframe for model input — assumes pipeline expects columns with these exact names
                input_df = pd.DataFrame([{ 'Dependents': Dependents,
                                           'tenure': int(tenure),
                                           'OnlineSecurity': OnlineSecurity,
                                           'OnlineBackup': OnlineBackup,
                                           'InternetService': InternetService,
                                           'DeviceProtection': DeviceProtection,
                                           'TechSupport': TechSupport,
                                           'Contract': Contract,
                                           'PaperlessBilling': PaperlessBilling,
                                           'MonthlyCharges': float(MonthlyCharges)
                                         }])
                # verify model supports predict_proba
                try:
                    proba = model.predict_proba(input_df)
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                else:
                    # assume class order ['No','Yes'] or ['Not Churn','Churn'] — try to find churn column index
                    classes = None
                    try:
                        classes = model.classes_
                    except Exception:
                        # if pipeline, try named steps
                        try:
                            classes = model[-1].classes_
                        except Exception:
                            classes = None
                    # find index of positive class (something like 'Yes' or 'Churn')
                    pos_idx = 1
                    if classes is not None:
                        # choose most-likely positive label containing 'Yes' or 'Churn'
                        for i,c in enumerate(classes):
                            if str(c).lower() in ['yes','churn','1','true']:
                                pos_idx = i
                        # if not found keep 1
                    churn_prob = float(proba[0][pos_idx])
                    churn_pct = round(churn_prob*100,1)
                    pred_label = 'Churn' if churn_prob>=0.5 else 'Not Churn'

                    # confidence badge
                    if 0.4 <= churn_prob <= 0.6:
                        conf_msg = 'Low confidence (near decision boundary)'
                        st.warning(conf_msg)
                    else:
                        st.success(f'Prediction: {pred_label}')

                    st.markdown(f"**Probability churn:** {churn_pct}%")

                    # small gauge using plotly
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_prob*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': ACCENT}},
                        title={'text': "Churn probability (%)"}
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    # save to log
                    record = {'timestamp': datetime.utcnow().isoformat(),
                              **{k: input_df.iloc[0][k] for k in input_df.columns},
                              'prediction': pred_label,
                              'prob_churn': churn_prob}
                    try:
                        append_prediction_log(record)
                        st.info('Prediction saved to data_telco_customer_churn.csv')
                    except Exception as e:
                        st.error(f'Failed to save log: {e}')

# --------------------------
# Predict from file (batch)
# --------------------------
elif menu == "Predict from file":
    st.header("Batch prediction from CSV")
    st.markdown("Upload CSV with required features. The app will append `prediction` and `prob_churn` columns.")
    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'])
    if uploaded_file is None:
        st.info('Upload a CSV file to run batch predictions.')
    else:
        try:
            df_batch = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
            df_batch = None
        if df_batch is not None:
            st.write('Preview of uploaded file')
            st.write(df_batch.head())
            required = ['Dependents','tenure','OnlineSecurity','OnlineBackup','InternetService','DeviceProtection','TechSupport','Contract','PaperlessBilling','MonthlyCharges']
            missing = ensure_columns_for_model(df_batch, required)
            if missing:
                st.error(f"Uploaded CSV is missing required columns: {missing}")
            elif model is None:
                st.error("Model not loaded — cannot run batch prediction.")
            else:
                try:
                    probs = model.predict_proba(df_batch)
                except Exception as e:
                    st.error(f"Model predict_proba failed: {e}")
                else:
                    classes = None
                    try:
                        classes = model.classes_
                    except Exception:
                        try:
                            classes = model[-1].classes_
                        except Exception:
                            classes = None
                    pos_idx = 1
                    if classes is not None:
                        for i,c in enumerate(classes):
                            if str(c).lower() in ['yes','churn','1','true']:
                                pos_idx = i
                    prob_churn = probs[:, pos_idx]
                    pred_label = np.where(prob_churn>=0.5, 'Churn', 'Not Churn')
                    df_out = df_batch.copy()
                    df_out['prediction'] = pred_label
                    df_out['prob_churn'] = prob_churn

                    st.success('Batch prediction completed')
                    st.write(df_out.head())

                    # download
                    csv = df_out.to_csv(index=False).encode('utf-8')
                    st.download_button(label='Download results as CSV', data=csv, file_name='prediction_results.csv', mime='text/csv')

# --------------------------
# End of pages
# --------------------------

# Footer: quick requirements and notes
st.markdown('---')
with st.expander('Run instructions & requirements'):
    st.markdown('''
    **Run**: `pip install -r requirements.txt` then `streamlit run app.py`

    **requirements.txt** (example):

    ```text
    streamlit
    pandas
    numpy
    joblib
    scikit-learn
    plotly
    matplotlib
    ```
    ''')

# End
