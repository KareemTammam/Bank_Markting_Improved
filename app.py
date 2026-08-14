import streamlit as st

from style import inject_css
import pages_content as pc

st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

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
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.2rem;">
            <div style="font-size:1.7rem;">🏦</div>
            <div>
                <div style="font-weight:800;font-size:1.15rem;color:#0F172A;line-height:1.1;">Bank Marketing</div>
                <div style="font-size:0.75rem;color:#64748B;">Term Deposit Predictor</div>
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
        <div style="font-size:0.75rem;color:#94A3B8; line-height:1.6;">
        Built with scikit-learn, XGBoost &amp; Streamlit.<br>
        Predicts term-deposit subscription from a Portuguese bank's
        marketing campaign data.
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
