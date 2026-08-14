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

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# CSS is injected using whatever dark_mode value survived from the previous
# run; the toggle widget created later in the sidebar just updates that
# value and triggers a rerun — no ordering problem, since Streamlit reruns
# the whole script top to bottom on every interaction.
inject_css(dark=st.session_state.dark_mode)


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
    top_l, top_r = st.columns([2.4, 1])
    with top_l:
        st.markdown(
            "<div style='font-size:0.72rem; color:#94A3B8; padding-top:0.4rem;'>"
            "🌙 Dark mode for the main page</div>",
            unsafe_allow_html=True,
        )
    with top_r:
        st.toggle("Dark", key="dark_mode", label_visibility="collapsed")

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

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-muted" style="font-size:0.72rem; line-height:1.45;">
        Built with scikit-learn, XGBoost &amp; Streamlit — predicts term-deposit
        subscription from a Portuguese bank's marketing campaign data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown("###### 📦 Dataset Info")
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
