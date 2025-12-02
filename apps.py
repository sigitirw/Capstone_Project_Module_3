# app.py
# Streamlit single-file app that attempts to safely load a scikit-learn pickle
# and displays a small inline SVG "telco" logo on the home page.
#
# This file implements a conservative monkeypatching approach to handle
# the specific pickle error: "Can't get attribute '_RemainderColsList' on module ..."
# The monkeypatch is applied only if the attribute is missing, and a clear
# warning is shown to the user. Prefer reinstalling the original scikit-learn
# version used to create the pickle for a long-term fix.

import os
import importlib
import traceback
import joblib
import streamlit as st
from types import SimpleNamespace

# ----------------------------- Configuration -----------------------------
MODEL_FILENAME = "telcoChurn.pkl"
# If your app placed the model in a subfolder, set MODEL_PATH accordingly.
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)

st.set_page_config(page_title="Telco Churn App", page_icon="📶", layout="wide")

# ----------------------------- Helper functions -----------------------------

def ensure_remainder_placeholder():
    """Ensure the module sklearn.compose._column_transformer exposes
    the attribute _RemainderColsList that the pickle expects.

    This creates a lightweight placeholder class only when the attribute
    is missing. The placeholder tries to be permissive enough for unpickling
    to succeed but it does NOT guarantee correct runtime semantics.

    Use this only as a debugging / migration aid. Prefer installing the
    matching scikit-learn version used when serializing the model.
    """
    try:
        mod = importlib.import_module("sklearn.compose._column_transformer")
    except Exception:
        # Couldn't import the sklearn module at all; let the caller handle it.
        return False, "Failed to import sklearn.compose._column_transformer"

    if hasattr(mod, "_RemainderColsList"):
        return True, "Attribute already present"

    # Create a permissive placeholder class
    class _RemainderColsList:
        def __init__(self, *args, **kwargs):
            # store anything so unpickling that expects attributes doesn't fail
            self._args = args
            self._kwargs = kwargs

        def __repr__(self):
            return "<_RemainderColsList placeholder>"

    # Attach to the module so pickle can find it
    setattr(mod, "_RemainderColsList", _RemainderColsList)
    return True, "Placeholder _RemainderColsList installed"


def load_model_with_monkeypatch(path: str):
    """Try to load a joblib/pickle model.

    Strategy:
    1. Attempt a direct joblib.load.
    2. If it fails with an attribute-not-found related to _RemainderColsList,
       attempt the placeholder monkeypatch and retry.
    3. Return (model_or_none, error_message_or_None)
    """
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"

    # try load first
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        tb = traceback.format_exc()
        # Quick textual check for the specific attribute error mentioned by the user
        msg = str(e)
        if "_RemainderColsList" in msg or "Can't get attribute" in msg:
            # Attempt monkeypatch and retry
            patched, patch_msg = ensure_remainder_placeholder()
            try:
                model = joblib.load(path)
                return model, None
            except Exception as e2:
                return None, (
                    "Failed loading model even after installing placeholder. "
                    f"Patch info: {patch_msg}. Last error: {e2}
Full traceback:
{traceback.format_exc()}"
                )
        else:
            # Some other error - return full trace for debugging
            return None, f"Failed loading model. Error: {msg}
Traceback:
{tb}"


# ----------------------------- UI: Logo ----------------------------------

# Inline SVG logo representing a telco company (antenna + signal waves)
# This avoids external image dependencies and works offline.
TELCO_SVG = r"""
<div style="display:flex;align-items:center;gap:12px">
  <div style="width:84px;height:84px;background:linear-gradient(135deg,#e6f2ff,#ffffff);
              border-radius:18px;display:flex;align-items:center;justify-content:center;box-shadow:0 6px 18px rgba(0,0,0,0.06)">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="56" height="56">
      <defs>
        <linearGradient id="g1" x1="0" x2="1">
          <stop offset="0" stop-color="#0b6cff" />
          <stop offset="1" stop-color="#00c2ff" />
        </linearGradient>
      </defs>
      <!-- antenna mast -->
      <rect x="30" y="18" width="4" height="22" rx="2" fill="#0b6cff" />
      <!-- circle at top -->
      <circle cx="32" cy="14" r="4" fill="url(#g1)" />
      <!-- three signal arcs -->
      <path d="M20 36a12 12 0 0 1 24 0" fill="none" stroke="#0b6cff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.95" />
      <path d="M16 42a20 20 0 0 1 32 0" fill="none" stroke="#00c2ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity="0.9" />
      <path d="M12 48a28 28 0 0 1 40 0" fill="none" stroke="#66d9ff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.85" />
    </svg>
  </div>
  <div style="font-family:Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;">
    <div style="font-size:18px;font-weight:700;color:#05386b">Telco Churn Explorer</div>
    <div style="color:#2b6ca3;font-size:12px">Quick demo · verify model predictions before using in production</div>
  </div>
</div>
"""

# Show logo on the top-left of the app
st.markdown(TELCO_SVG, unsafe_allow_html=True)
st.write("---")

# ----------------------------- Main UI ----------------------------------
st.header("Model loader & diagnostics")

st.info(f"Attempting to load model file: **{MODEL_FILENAME}** from working directory: `{os.getcwd()}`")
st.write(f"Exists: {os.path.exists(MODEL_PATH)}")

# Try to load the model now
model, error = load_model_with_monkeypatch(MODEL_PATH)

if model is not None:
    st.success("Model loaded successfully. **Important**: verify predictions on known inputs!")
    # show model type information
    try:
        st.write("Model type:", type(model))
        # If it's a pipeline, summarize steps if possible
        if hasattr(model, "named_steps"):
            st.write("Pipeline named steps:")
            for name, step in model.named_steps.items():
                st.write(f"- {name}: {type(step)}")
    except Exception:
        # non-fatal
        pass

    # Small sample input form (user can provide one row of features as CSV-like)
    st.subheader("Quick test prediction (single-row CSV-like)")
    st.write("Enter a single row with comma-separated values for numeric features (or leave default).
This is a lightweight tester — adapt to your real feature names/format.")
    sample = st.text_input("Sample CSV row", value="")
    if st.button("Run prediction on sample"):
        if not sample.strip():
            st.warning("No sample provided. Please enter a CSV-like row (values only).")
        else:
            try:
                vals = [float(x.strip()) for x in sample.split(",")]
                import numpy as np
                X = np.array(vals).reshape(1, -1)
                pred = model.predict(X)
                # attempt to show predict_proba if available
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                st.write("Prediction:", pred)
                if proba is not None:
                    st.write("Predict_proba:")
                    st.write(proba)
            except Exception as e:
                st.error(f"Failed to run prediction on sample. Error: {e}")
                st.write(traceback.format_exc())

else:
    st.error("Model not loaded.")
    st.write("Details:")
    st.code(error or "Unknown error", language="text")

    st.markdown("**Suggested next steps:**")
    st.markdown("- Pastikan file `telcoChurn.pkl` berada di folder yang sama dengan file app ini, atau ubah `MODEL_PATH` ke absolute path.")
    st.markdown("- Ideal: gunakan versi scikit-learn yang sama seperti saat membuat pickle. Contoh: `pip install scikit-learn==1.2.2`.")
    st.markdown("- Jika tidak bisa menggunakan versi yang sama, app mencoba membuat placeholder `_RemainderColsList` untuk memungkinkan unpickle. "
                "Namun hasilnya perlu diverifikasi (prediksi uji). Ini adalah workaround, bukan solusi permanen.")

# Footer / diagnostics
st.write("---")
with st.expander("Environment diagnostics (helpful for debugging)"):
    try:
        import sklearn
        st.write(f"scikit-learn version: {sklearn.__version__}")
    except Exception:
        st.write("scikit-learn not importable in this environment")
    st.write(f"Python cwd: {os.getcwd()}")
    st.write(f"Model path: {MODEL_PATH}")
    st.write("Files in cwd:")
    try:
        files = os.listdir(os.getcwd())
        st.write(files)
    except Exception as e:
        st.write(f"Could not list cwd: {e}")

st.caption("Note: If the model was pickled with a different scikit-learn version, the most reliable fix is to re-create the environment with that version and reload the model.")
