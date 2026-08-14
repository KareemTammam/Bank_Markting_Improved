import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style import COLORS, MODEL_PALETTE, METRIC_COLORS, page_header
from utils import (
    build_raw_input_row,
    load_confusion_matrices,
    load_feature_importance,
    load_meta,
    load_model,
    load_model_comparison,
    load_preprocessor,
    load_raw_options,
    load_sample_data,
    load_target_correlation,
    prettify_feature,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    margin=dict(l=10, r=10, t=40, b=10),
)


def kpi_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# HOME
# =============================================================================
def render_home():
    meta = load_meta()
    comparison = load_model_comparison()
    best = comparison.iloc[0]

    left, right = st.columns([1.3, 1])
    with left:
        # All hero content must be emitted in a SINGLE st.markdown call —
        # Streamlit renders each st.markdown as its own isolated DOM
        # container, so a <div> opened in one call and closed in another
        # does NOT wrap the elements in between (the browser auto-closes
        # it), leaving an empty colored box floating above the text.
        st.markdown(
            """
            <div class="hero-frame">
                <div class="badge badge-blue">Machine Learning · Binary Classification</div>
                <div class="hero-title">Bank Marketing Prediction</div>
                <div class="hero-sub">Predicts whether a bank customer will subscribe to a
                term deposit after a marketing call — trained on real Portuguese bank
                telemarketing campaign data and benchmarked across 8 ML models.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("🎯  Make a Prediction", width='stretch'):
                st.session_state.nav_request = "Make a Prediction"
                st.rerun()
        with c2:
            if st.button("📊  View Model Dashboard", width='stretch'):
                st.session_state.nav_request = "Model Dashboard"
                st.rerun()

    with right:
        st.markdown(
            f"""
            <div class="card">
                <div class="kpi-label">Best Model</div>
                <div class="kpi-value" style="font-size:1.6rem;">{best['Model']}</div>
                <hr style="margin:0.9rem 0;">
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <div class="kpi-label">F1 Score</div>
                        <div class="kpi-value" style="font-size:1.4rem; color:{COLORS['primary']};">{best['F1']*100:.1f}%</div>
                    </div>
                    <div>
                        <div class="kpi-label">Accuracy</div>
                        <div class="kpi-value" style="font-size:1.4rem; color:{COLORS['success']};">{best['Accuracy']*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    st.markdown("#### What this project does")
    st.markdown(
        "<p class='muted'>A Portuguese bank runs phone-call marketing campaigns to sell term "
        "deposits. This tool learns from over 11,000 past campaign contacts — customer "
        "demographics, account details, and call history — to predict whether a new "
        "customer is likely to say <b>yes</b>, helping marketing teams prioritize outreach.</p>",
        unsafe_allow_html=True,
    )

    st.write("")
    cols = st.columns(4)
    stats = [
        ("Training Samples", f"{meta['n_samples']:,}", "Historical campaign contacts"),
        ("Engineered Features", f"{meta['n_features_encoded']}", f"From {meta['n_features_raw']} raw fields"),
        ("Models Benchmarked", "8", "Classical ML to gradient boosting"),
        ("Best F1 Score", f"{best['F1']*100:.1f}%", f"Achieved by {best['Model']}"),
    ]
    for col, (label, val, sub) in zip(cols, stats):
        with col:
            kpi_card(label, val, sub)

    st.write("")
    st.write("")
    st.markdown("#### How it works")
    steps = [
        ("Enter customer details", "Demographics, financial profile, and campaign contact info."),
        ("Model processes the input", "The same encoding & scaling pipeline used in training is applied instantly."),
        ("Get an instant prediction", "See the likely outcome with a confidence score, in a clear result card."),
    ]
    scols = st.columns(3)
    for i, (col, (title, desc)) in enumerate(zip(scols, steps)):
        with col:
            st.markdown(
                f"""
                <div class="card" style="min-height:130px;">
                    <span class="step-num">{i+1}</span><b>{title}</b>
                    <p class="muted" style="margin-top:0.5rem; font-size:0.9rem;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# DATASET
# =============================================================================
def render_dataset():
    meta = load_meta()
    sample = load_sample_data()
    corr = load_target_correlation()

    page_header(
        "Dataset Overview",
        "Direct marketing (phone call) campaign records from a Portuguese banking "
        "institution. The classification target is whether the client subscribed to "
        "a term deposit.",
        icon="🗂️",
    )

    cols = st.columns(4)
    class_counts = meta["class_counts"]
    total = sum(class_counts.values())
    stats = [
        ("Total Samples", f"{meta['n_samples']:,}", ""),
        ("Raw Features", f"{meta['n_features_raw']}", "Before encoding"),
        ("Train / Test Split", f"{meta['n_train']:,} / {meta['n_test']:,}", "80 / 20 stratified"),
        ("Class Balance", f"{class_counts.get('1', class_counts.get(1, 0))/total*100:.1f}% Yes",
         f"{class_counts.get('0', class_counts.get(0, 0))/total*100:.1f}% No"),
    ]
    for col, (label, val, sub) in zip(cols, stats):
        with col:
            kpi_card(label, val, sub)

    st.write("")
    left, right = st.columns([1, 1.3])

    with left:
        st.markdown("##### Target Distribution")
        counts = {"No": class_counts.get("0", class_counts.get(0, 0)),
                  "Yes": class_counts.get("1", class_counts.get(1, 0))}
        fig = go.Figure(
            go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.62,
                marker=dict(colors=[COLORS["warning"], COLORS["success"]]),
                textinfo="label+percent",
                textfont=dict(size=13, family="Inter"),
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            showlegend=False,
            annotations=[dict(text="Deposit", x=0.5, y=0.5, font_size=16, showarrow=False,
                               font=dict(color=COLORS["text"], family="Inter"))],
        )
        st.plotly_chart(fig, width='stretch')

    with right:
        st.markdown("##### Top Features Correlated with Subscription")
        top_corr = corr.head(10).sort_values()
        bar_colors = [COLORS["success"] if v > 0 else COLORS["danger"] for v in top_corr.values]
        fig2 = go.Figure(
            go.Bar(
                x=top_corr.values,
                y=[prettify_feature(f) for f in top_corr.index],
                orientation="h",
                marker_color=bar_colors,
            )
        )
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            xaxis_title="Correlation with target",
            yaxis=dict(automargin=True),
        )
        st.plotly_chart(fig2, width='stretch')

    st.write("")
    st.markdown("##### Sample of the Data")
    st.dataframe(sample.head(12), width='stretch', hide_index=True)

    st.write("")
    st.markdown("##### Feature Reference")
    feat_info = pd.DataFrame(
        [
            ("age", "Numeric", "Client's age"),
            ("job", "Categorical", "Type of job (12 categories)"),
            ("marital", "Categorical", "Marital status"),
            ("education", "Ordinal", "Primary → Secondary → Tertiary → Unknown"),
            ("default", "Binary", "Has credit in default?"),
            ("balance", "Numeric", "Average yearly balance (EUR)"),
            ("housing", "Binary", "Has a housing loan?"),
            ("loan", "Binary", "Has a personal loan?"),
            ("contact", "Categorical", "Contact communication type"),
            ("day / month", "Numeric / Categorical", "Last contact date"),
            ("duration", "Numeric", "Last contact duration (seconds)"),
            ("campaign", "Numeric", "# of contacts during this campaign"),
            ("pdays", "Numeric", "Days since last contact in a previous campaign"),
            ("poutcome", "Categorical", "Outcome of the previous campaign"),
        ],
        columns=["Feature", "Type", "Description"],
    )
    st.dataframe(feat_info, width='stretch', hide_index=True)


# =============================================================================
# MODEL DASHBOARD
# =============================================================================
def render_dashboard():
    comparison = load_model_comparison()
    fi = load_feature_importance()
    cms = load_confusion_matrices()
    best = comparison.iloc[0]

    page_header(
        "Model Performance Dashboard",
        "8 classification models were trained and tuned via grid search cross-validation, "
        "then evaluated on a held-out test set.",
        icon="🤖",
    )

    cols = st.columns(4)
    kpis = [
        ("Best Model", best["Model"], "Highest F1 score"),
        ("Accuracy", f"{best['Accuracy']*100:.1f}%", ""),
        ("Precision", f"{best['Precision']*100:.1f}%", ""),
        ("Recall", f"{best['Recall']*100:.1f}%", ""),
    ]
    for col, (label, val, sub) in zip(cols, kpis):
        with col:
            kpi_card(label, val, sub)

    st.write("")
    st.markdown("##### Model Comparison — Accuracy, Precision, Recall, F1")
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=comparison["Model"],
                y=comparison[metric],
                marker_color=METRIC_COLORS[metric],
                text=[f"{v:.2f}" for v in comparison[metric]],
                textposition="outside",
                textfont=dict(size=10),
            )
        )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        height=460,
        yaxis=dict(range=[0, 1.08], title="Score", gridcolor="#F1F5F9"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
        bargap=0.25,
        bargroupgap=0.08,
    )
    st.plotly_chart(fig, width='stretch')

    st.write("")
    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"##### 🧮 Confusion Matrix — {best['Model']}")
        model_names = list(cms.keys())
        default_idx = model_names.index(best["Model"]) if best["Model"] in model_names else 0
        chosen = st.selectbox("Select a model", model_names, index=default_idx, key="cm_select")
        cm = np.array(cms[chosen])
        labels = ["No", "Yes"]
        cm_max = cm.max() if cm.max() > 0 else 1
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale=[[0, "#CFE0FB"], [1, COLORS["primary"]]],
                zmin=0,
                zmax=cm_max,
                showscale=False,
            )
        )
        # Plotly's Heatmap textfont.color only accepts a single color, not
        # a per-cell array, so per-cell readable text (white on the dark
        # cells, dark navy on the light cells) is added as annotations
        # instead of texttemplate.
        for i, row_label in enumerate(labels):
            for j, col_label in enumerate(labels):
                value = int(cm[i][j])
                text_color = "#FFFFFF" if value >= cm_max * 0.45 else COLORS["text"]
                fig_cm.add_annotation(
                    x=col_label, y=row_label, text=str(value),
                    showarrow=False, font=dict(size=18, color=text_color),
                )
        fig_cm.update_layout(
            **PLOTLY_LAYOUT,
            height=340,
            xaxis=dict(title="Predicted", side="bottom"),
            yaxis=dict(title="Actual", autorange="reversed"),
        )
        st.plotly_chart(fig_cm, width='stretch')

    with right:
        st.markdown(f"##### Top Features — {best['Model']}")
        fi_sorted = fi.sort_values("importance").tail(10)
        fig_fi = go.Figure(
            go.Bar(
                x=fi_sorted["importance"],
                y=[prettify_feature(f) for f in fi_sorted["feature"]],
                orientation="h",
                marker_color=COLORS["primary"],
            )
        )
        fig_fi.update_layout(
            **PLOTLY_LAYOUT,
            height=340,
            xaxis_title="Importance",
            yaxis=dict(automargin=True),
        )
        st.plotly_chart(fig_fi, width='stretch')

    st.write("")
    st.markdown("##### Full Results Table")
    display_df = comparison.copy()
    for m in metrics:
        display_df[m] = (display_df[m] * 100).round(2).astype(str) + "%"
    st.dataframe(display_df, width='stretch', hide_index=True)


# =============================================================================
# PREDICTION
# =============================================================================
def render_predict():
    model = load_model()
    preprocessor = load_preprocessor()
    options = load_raw_options()

    page_header(
        "Make a Prediction",
        "Fill in the customer's profile and campaign details below. The exact "
        "preprocessing pipeline used during training is applied automatically.",
        icon="🔮",
    )
    st.write("")

    with st.form("prediction_form"):
        st.markdown("##### 👤 Personal Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=95, value=41, step=1)
        with c2:
            job = st.selectbox("Job", options["job"], index=options["job"].index("management") if "management" in options["job"] else 0)
        with c3:
            marital = st.selectbox("Marital Status", options["marital"], index=options["marital"].index("married") if "married" in options["marital"] else 0)

        education = st.select_slider(
            "Education Level",
            options=["primary", "secondary", "tertiary", "unknown"],
            value="secondary",
        )

        st.write("")
        st.markdown("##### 💰 Financial Profile")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            balance = st.number_input("Account Balance (EUR)", min_value=-8000, max_value=100000, value=1500, step=100)
        with c5:
            default = st.selectbox("Credit in Default?", ["no", "yes"])
        with c6:
            housing = st.selectbox("Housing Loan?", ["no", "yes"], index=1)
        with c7:
            loan = st.selectbox("Personal Loan?", ["no", "yes"])

        st.write("")
        st.markdown("##### 📞 Campaign Contact Details")
        c8, c9, c10 = st.columns(3)
        with c8:
            contact = st.selectbox("Contact Type", options["contact"])
        with c9:
            month = st.selectbox("Last Contact Month", options["month"], index=options["month"].index("may"))
        with c10:
            day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15, step=1)

        c11, c12 = st.columns(2)
        with c11:
            duration = st.number_input(
                "Last Call Duration (seconds)", min_value=0, max_value=5000, value=180, step=10,
                help="Strongest predictor — longer calls correlate heavily with subscription.",
            )
        with c12:
            campaign = st.number_input("Contacts During This Campaign", min_value=1, max_value=60, value=2, step=1)

        st.write("")
        st.markdown("##### 📈 Previous Campaign History")
        c13, c14 = st.columns(2)
        with c13:
            pdays = st.number_input(
                "Days Since Previous Contact", min_value=-1, max_value=900, value=-1, step=1,
                help="-1 means the client was not previously contacted.",
            )
        with c14:
            poutcome = st.selectbox("Previous Campaign Outcome", options["poutcome"])

        st.write("")
        submitted = st.form_submit_button("🔮  Predict Subscription", width='stretch')

    if submitted:
        with st.spinner("Running the model..."):
            form_values = {
                "age": age, "job": job, "marital": marital, "education": education,
                "default": 1 if default == "yes" else 0,
                "balance": balance,
                "housing": 1 if housing == "yes" else 0,
                "loan": 1 if loan == "yes" else 0,
                "contact": contact, "day": day, "month": month, "duration": duration,
                "campaign": campaign, "pdays": pdays, "poutcome": poutcome,
            }
            raw_row = build_raw_input_row(form_values)
            try:
                encoded = preprocessor.transform(raw_row)
                proba = model.predict_proba(encoded)[0]
                pred = int(np.argmax(proba))
                confidence = proba[pred]
            except Exception as e:
                st.error(f"Something went wrong while running the prediction: {e}")
                return

        st.write("")
        result_class = "result-yes" if pred == 1 else "result-no"
        result_text = "WILL SUBSCRIBE ✅" if pred == 1 else "UNLIKELY TO SUBSCRIBE ✕"
        explanation = (
            "Based on this profile and call pattern, the model finds a strong signal "
            "consistent with customers who subscribed to a term deposit."
            if pred == 1 else
            "Based on this profile and call pattern, the model finds this customer "
            "resembles those who did not subscribe."
        )

        rc1, rc2 = st.columns([1.4, 1])
        with rc1:
            st.markdown(
                f"""
                <div class="result-card {result_class}">
                    <div class="result-label">Prediction Result</div>
                    <div class="result-value">{result_text}</div>
                    <div class="muted" style="font-size:0.92rem;">{explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rc2:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    number={"suffix": "%", "font": {"size": 34}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": COLORS["success"] if pred == 1 else COLORS["warning"]},
                        "bgcolor": "white",
                        "borderwidth": 0,
                    },
                    title={"text": "Confidence Score", "font": {"size": 14}},
                )
            )
            gauge_layout = {**PLOTLY_LAYOUT, "margin": dict(l=20, r=20, t=40, b=10)}
            fig.update_layout(**gauge_layout, height=230)
            st.plotly_chart(fig, width='stretch')

        st.write("")
        st.markdown("##### Class Probabilities")
        p1, p2 = st.columns(2)
        with p1:
            st.progress(float(proba[0]), text=f"No — {proba[0]*100:.1f}%")
        with p2:
            st.progress(float(proba[1]), text=f"Yes — {proba[1]*100:.1f}%")


# =============================================================================
# ABOUT
# =============================================================================
def render_about():
    comparison = load_model_comparison()
    best = comparison.iloc[0]

    page_header("About This Project", icon="ℹ️")

    st.markdown("##### 🎯 Problem")
    st.markdown(
        "<p class='muted'>A Portuguese bank runs outbound phone-call marketing campaigns to "
        "sell term deposit products. Contacting every customer is costly and time-consuming, "
        "so the bank wants to predict — before or during a campaign — which customers are "
        "most likely to subscribe, allowing marketing teams to prioritize their efforts.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("##### 🗂️ Dataset")
    st.markdown(
        "<p class='muted'>11,162 records of past campaign contacts, each with 17 fields "
        "covering client demographics (age, job, marital status, education), financial "
        "profile (account balance, existing loans, credit default), and campaign contact "
        "history (contact type, timing, duration, previous outcomes). The target variable "
        "<b>deposit</b> indicates whether the client subscribed.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("##### 🛠️ Preprocessing")
    steps = [
        "Dropped the `previous` column and mapped the target `deposit` to 0/1 once, used consistently downstream.",
        "Binary fields (`default`, `housing`, `loan`) mapped to 0/1.",
        "`education` ordinally encoded (primary → secondary → tertiary → unknown).",
        "`job`, `marital`, `contact`, `month`, `poutcome` one-hot encoded.",
        "Numeric fields (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`) standardized with `StandardScaler`.",
        "Stratified 80/20 train-test split, fit once and reused for every model.",
    ]
    for s in steps:
        st.markdown(f"<div style='margin-bottom:0.4rem;'>• <span class='muted'>{s}</span></div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("##### 🤖 Models Tested")
    model_list = ", ".join(comparison["Model"].tolist())
    st.markdown(
        f"<p class='muted'>Eight classifiers were trained and hyperparameter-tuned with "
        f"5-fold stratified cross-validation (optimizing for F1 score): {model_list}.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("##### 📏 Evaluation Metrics")
    metric_cards = st.columns(4)
    metric_defs = [
        ("Accuracy", "Overall share of correct predictions."),
        ("Precision", "Of predicted subscribers, how many actually subscribed."),
        ("Recall", "Of actual subscribers, how many were correctly identified."),
        ("F1 Score", "Harmonic mean of precision and recall — the primary model-selection metric."),
    ]
    for col, (name, desc) in zip(metric_cards, metric_defs):
        with col:
            st.markdown(
                f"""<div class="card" style="min-height:130px;">
                <b>{name}</b>
                <p class="muted" style="font-size:0.85rem; margin-top:0.4rem;">{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("##### 🏆 Best Model")
    st.markdown(
        f"""
        <div class="card">
        <span class="badge badge-green">Selected</span>
        <h4 style="margin-top:0.6rem;">{best['Model']}</h4>
        <p class="muted">
        {best['Model']} achieved the highest F1 score ({best['F1']*100:.1f}%) among all
        candidates, with {best['Accuracy']*100:.1f}% accuracy, {best['Precision']*100:.1f}%
        precision, and {best['Recall']*100:.1f}% recall on the held-out test set. Gradient-boosted
        tree ensembles like this one tend to outperform simpler models here because they capture
        non-linear interactions between call duration, previous campaign outcomes, and contact
        method — the strongest signals in this dataset.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("##### 🔗 Tech Stack")
    st.markdown(
        "<p class='muted'>Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly</p>",
        unsafe_allow_html=True,
    )
