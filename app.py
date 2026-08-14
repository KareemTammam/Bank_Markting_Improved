import streamlit as st

from style import inject_css
import pages_content as pc
import utils

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


@st.cache_data(show_spinner=False)
def _dataset_shape():
    """Real (rows, columns) of the source CSV. Falls back to the numbers
    stored in meta.json (from the training run) if the raw file isn't
    bundled with this deployment."""
    try:
        import pandas as pd
        from pathlib import Path
        df = pd.read_csv(Path(__file__).parent / "data" / "bank.csv")
        return df.shape[0], df.shape[1]
    except FileNotFoundError:
        meta = utils.load_meta()
        return meta["n_samples"], meta["n_features_raw"] + 1  # +1 for target column


PAGES = {
    "Overview": "home",
    "Dataset": "dataset",
    "Model Dashboard": "dashboard",
    "Make a Prediction": "predict",
    "About the Project": "about",
}

if "nav" not in st.session_state:
    st.session_state.nav = "Overview"

# Apply any pending navigation request (set by in-page buttons) BEFORE the
# sidebar radio widget below is instantiated — Streamlit forbids mutating a
# widget-bound session_state key after that widget has been created.
if "nav_request" in st.session_state:
    st.session_state.nav = st.session_state.pop("nav_request")

with st.sidebar:
    st.markdown(
        """
        <div class="brand-frame">
            <div style="font-size:1.7rem;">🏦</div>
            <div>
                <div class="brand-title">Bank Marketing<br>Prediction</div>
                <div class="brand-sub">Term Deposit Predictor</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selection = st.radio(
        "Navigate",
        list(PAGES.keys()),
        label_visibility="collapsed",
        key="nav",
    )

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:0.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-muted" style="font-size:0.75rem; line-height:1.6;">
        Built with scikit-learn, XGBoost &amp; Streamlit.<br>
        Predicts term-deposit subscription from a Portuguese bank's
        marketing campaign data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:0.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("##### 📦 Dataset Info")
    n_rows, n_cols = _dataset_shape()
    st.markdown(
        f"""
        <div class="sidebar-stat-row">
            <span class="sidebar-stat-label">Rows</span>
            <span class="sidebar-stat-value">{n_rows:,}</span>
        </div>
        <div class="sidebar-stat-row">
            <span class="sidebar-stat-label">Columns</span>
            <span class="sidebar-stat-value">{n_cols}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dev-signature-wrap">
            <div class="dev-signature">Kareem Tammam</div>
            <div class="dev-signature-sub">Crafted with passion</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

page_key = PAGES[selection]

if page_key == "home":
    pc.render_home()
elif page_key == "dataset":
    pc.render_dataset()
elif page_key == "dashboard":
    pc.render_dashboard()
elif page_key == "predict":
    pc.render_predict()
elif page_key == "about":
    pc.render_about()
