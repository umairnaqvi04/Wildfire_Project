"""
Wildfire Prediction System — Streamlit App
--------------------------------------------
Single-file deployment on purpose: avoids any folder/path issues on
Streamlit Cloud. "Pages" are simulated with a sidebar radio selector instead
of st.Page()/pages/ — same modern multi-page feel, zero extra files needed.

Run locally:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import backend as bk

# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Wildfire Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg_card": "#211a14",
    "bg_card_alt": "#2a2117",
    "border": "#3a2e22",
    "ember": "#ff6b35",
    "amber": "#ffba08",
    "deep_red": "#c1121f",
    "smoke": "#9c958b",
    "safe": "#5e7a72",
    "text": "#f2e9e4",
    "text_dim": "#cdc3b8",
}

st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{
        background: radial-gradient(circle at 15% 0%, #2a1d12 0%, #15110d 55%);
        color: {COLORS['text']};
    }}
    section[data-testid="stSidebar"] {{
        background: {COLORS['bg_card']};
        border-right: 1px solid {COLORS['border']};
    }}
    h1, h2, h3 {{ font-family: 'Oswald', sans-serif; letter-spacing: 0.02em; color: {COLORS['text']}; }}
    h1 {{ text-transform: uppercase; border-bottom: 3px solid {COLORS['ember']}; padding-bottom: 0.3rem; display: inline-block; }}
    p, li, label, .stMarkdown {{ color: {COLORS['text_dim']}; }}
    div[data-testid="stMetric"] {{
        background: {COLORS['bg_card']}; border: 1px solid {COLORS['border']};
        border-left: 4px solid {COLORS['ember']}; border-radius: 6px; padding: 1rem 1.2rem;
    }}
    div[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; color: {COLORS['ember']}; }}
    div[data-testid="stMetricLabel"] {{ color: {COLORS['smoke']}; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.08em; }}
    .stButton > button, .stFormSubmitButton > button {{
        background: linear-gradient(135deg, {COLORS['ember']} 0%, {COLORS['deep_red']} 100%);
        color: #15110d; font-weight: 600; border: none; border-radius: 6px;
        padding: 0.6rem 1.4rem; text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .ember-card {{
        background: {COLORS['bg_card']}; border: 1px solid {COLORS['border']};
        border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    }}
    .ember-eyebrow {{
        color: {COLORS['ember']}; font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = dict(
    paper_bgcolor=COLORS["bg_card"], plot_bgcolor=COLORS["bg_card"],
    font=dict(color=COLORS["text_dim"], family="Inter"),
)


def risk_color(label):
    return {"Low Risk": COLORS["safe"], "Moderate Risk": COLORS["amber"], "High Risk": COLORS["deep_red"]}.get(label, COLORS["smoke"])


# ----------------------------------------------------------------------
# Cached data + model loading (trains once per deployment, then cached)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    return bk.load_data("forestfires.csv")


@st.cache_resource(show_spinner="Training models (first run only)...")
def get_trained_bundle(_df):
    return bk.train_all(_df)


df = get_data()
bundle = get_trained_bundle(df)
pipelines = bundle["pipelines"]
metrics = bundle["metrics"]
kmeans = bundle["kmeans"]
cluster_scaler = bundle["cluster_scaler"]
risk_labels = bundle["risk_labels"]
cluster_profile = bundle["cluster_profile"]
X_test, y_test = bundle["X_test"], bundle["y_test"]

# ----------------------------------------------------------------------
# Sidebar navigation (simulates multi-page app, single file)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.2rem 0 1rem 0;">
        <span style="font-family:'Oswald',sans-serif;font-size:1.3rem;
        text-transform:uppercase;color:#f2e9e4;">🔥 Wildfire System</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["Home", "Data Explorer", "Model Comparison", "Live Prediction", "Risk Clusters"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("3 classifiers · 1 clustering model · trained on app start")

# ========================================================================
# PAGE: HOME
# ========================================================================
if page == "Home":
    st.markdown('<div class="ember-eyebrow">AIC-221 · INTRODUCTION TO MACHINE LEARNING · IQRA UNIVERSITY</div>', unsafe_allow_html=True)
    st.title("Wildfire Prediction System")
    st.markdown(
        "Predicting whether weather and fire-index conditions in Portugal's northeast "
        "forests are likely to produce **significant burn damage**, using the UCI "
        "Forest Fires dataset."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Incident Records", f"{len(df)}")
    with c2:
        st.metric("Environmental Features", "10")
    with c3:
        st.metric("High-Damage Rate", f"{round(df['fire_risk'].mean()*100,1)}%")
    with c4:
        st.metric("Models Trained", "4")

    st.markdown("###")
    st.subheader("Problem Statement")
    st.markdown(
        """
        <div class="ember-card">
        Each record corresponds to a forest fire reported in the Montesinho natural
        park region. Alongside spatial coordinates and the date, it carries four
        <b>FWI (Fire Weather Index)</b> components — FFMC, DMC, DC, ISI — plus direct
        weather readings (temperature, humidity, wind, rain) and the final burned
        <b>area</b> in hectares. We treat <code>area &gt; 0</code> as a fire that
        caused measurable damage, and <code>area = 0</code> as one contained with
        negligible loss — a binary risk-classification problem.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Pipeline")
    steps = [
        ("01", "Collect", "517 records, 13 raw attributes from the UCI repository (Kaggle mirror)."),
        ("02", "Preprocess", "One-hot encode month/day, standardize 10 numeric features."),
        ("03", "Train", "Random Forest, SVM, KNN classifiers + K-Means clustering."),
        ("04", "Evaluate & Predict", "Accuracy/Precision/Recall/F1, then live single-case inference."),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="ember-card" style="min-height:150px;">
                <div class="ember-eyebrow">{num}</div>
                <div style="font-family:'Oswald',sans-serif;font-size:1.1rem;margin:0.3rem 0;color:{COLORS['text']};">{title}</div>
                <div style="font-size:0.85rem;color:{COLORS['smoke']};">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.info("Use the sidebar to explore the data, compare models, run a live prediction, or inspect risk clusters.", icon="🧭")

# ========================================================================
# PAGE: DATA EXPLORER
# ========================================================================
elif page == "Data Explorer":
    st.markdown('<div class="ember-eyebrow">EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)
    st.title("Data Explorer")

    tab1, tab2, tab3 = st.tabs(["Overview", "Distributions", "Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sample records**")
            st.dataframe(df.drop(columns=["fire_risk"]).head(12), height=420, width="stretch")
        with c2:
            st.markdown("**Fire-risk balance**")
            counts = df["fire_risk"].value_counts().rename({0: "Low Risk", 1: "High Risk"})
            fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55,
                                    marker=dict(colors=[COLORS["safe"], COLORS["deep_red"]])))
            fig.update_layout(**PLOTLY_TEMPLATE, height=380)
            st.plotly_chart(fig, width="stretch")

            st.markdown("**Records by month**")
            month_counts = df["month"].value_counts().reindex(bk.MONTHS).fillna(0)
            fig2 = go.Figure(go.Bar(x=month_counts.index, y=month_counts.values, marker_color=COLORS["ember"]))
            fig2.update_layout(**PLOTLY_TEMPLATE, height=300)
            st.plotly_chart(fig2, width="stretch")

    with tab2:
        numeric_cols = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
        feature = st.selectbox("Feature", numeric_cols, index=4)
        fig3 = px.histogram(df, x=feature, color=df["fire_risk"].map({0: "Low Risk", 1: "High Risk"}),
                             nbins=30, color_discrete_map={"Low Risk": COLORS["safe"], "High Risk": COLORS["deep_red"]},
                             opacity=0.75, barmode="overlay")
        fig3.update_layout(**PLOTLY_TEMPLATE, height=420)
        st.plotly_chart(fig3, width="stretch")
        st.caption("Overlapping distributions are why this is a genuinely hard classification problem.")

    with tab3:
        corr_cols = ["X","Y","FFMC","DMC","DC","ISI","temp","RH","wind","rain","fire_risk"]
        corr = df[corr_cols].corr()
        fig4 = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                     colorscale=[[0, COLORS["safe"]], [0.5, COLORS["bg_card_alt"]], [1, COLORS["deep_red"]]],
                                     zmin=-1, zmax=1))
        fig4.update_layout(**PLOTLY_TEMPLATE, height=480)
        st.plotly_chart(fig4, width="stretch")

# ========================================================================
# PAGE: MODEL COMPARISON
# ========================================================================
elif page == "Model Comparison":
    st.markdown('<div class="ember-eyebrow">SUPERVISED MODEL EVALUATION</div>', unsafe_allow_html=True)
    st.title("Model Comparison")

    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.markdown(f"**{m['model']}**")
            st.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
            st.metric("F1-score", f"{m['f1_score']*100:.1f}%")

    st.markdown("###")
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    fig = go.Figure()
    colors = [COLORS["ember"], COLORS["amber"], COLORS["deep_red"]]
    for i, m in enumerate(metrics):
        fig.add_trace(go.Bar(name=m["model"], x=[k.replace("_"," ").title() for k in metric_keys],
                              y=[m[k] for k in metric_keys], marker_color=colors[i % len(colors)]))
    fig.update_layout(**PLOTLY_TEMPLATE, barmode="group", height=420, yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Confusion matrices")
    cm_cols = st.columns(len(metrics))
    for col, m in zip(cm_cols, metrics):
        with col:
            cm = np.array(m["confusion_matrix"])
            fig_cm = go.Figure(go.Heatmap(z=cm, x=["Pred: Low","Pred: High"], y=["True: Low","True: High"],
                                           colorscale=[[0, COLORS["bg_card_alt"]], [1, COLORS["ember"]]],
                                           showscale=False, text=cm, texttemplate="%{text}",
                                           textfont=dict(size=18, color=COLORS["text"])))
            fig_cm.update_layout(**PLOTLY_TEMPLATE, height=300, title=m["model"])
            st.plotly_chart(fig_cm, width="stretch")

    best_acc = max(metrics, key=lambda m: m["accuracy"])
    best_recall = max(metrics, key=lambda m: m["recall"])
    st.markdown(
        f"""
        <div class="ember-card">
        <b>{best_acc['model']}</b> achieves the highest accuracy ({best_acc['accuracy']*100:.1f}%).
        <b>{best_recall['model']}</b> has the strongest recall ({best_recall['recall']*100:.1f}%) —
        it misses the fewest actual high-damage fires, which matters more than raw accuracy
        in an early-warning context. All three land in a 60–70% band, consistent with
        published results: weather alone only partially explains realized burn damage.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ========================================================================
# PAGE: LIVE PREDICTION
# ========================================================================
elif page == "Live Prediction":
    st.markdown('<div class="ember-eyebrow">SINGLE-CASE INFERENCE</div>', unsafe_allow_html=True)
    st.title("Live Prediction")
    st.markdown("Enter conditions for one location and moment in time — every classifier scores it independently.")

    with st.form("prediction_form"):
        st.markdown("##### Location & date")
        c1, c2, c3, c4 = st.columns(4)
        with c1: x_coord = st.slider("X coordinate", 1, 9, 5)
        with c2: y_coord = st.slider("Y coordinate", 1, 9, 5)
        with c3: month = st.selectbox("Month", bk.MONTHS, index=7)
        with c4: day = st.selectbox("Day", bk.DAYS, index=4)

        st.markdown("##### Fire Weather Index components")
        c5, c6, c7, c8 = st.columns(4)
        with c5: ffmc = st.slider("FFMC", 18.7, 96.2, 90.0)
        with c6: dmc = st.slider("DMC", 1.1, 291.3, 110.0)
        with c7: dc = st.slider("DC", 7.9, 860.6, 550.0)
        with c8: isi = st.slider("ISI", 0.0, 56.1, 9.0)

        st.markdown("##### Weather readings")
        c9, c10, c11, c12 = st.columns(4)
        with c9: temp = st.slider("Temperature (°C)", 2.2, 33.3, 19.0)
        with c10: rh = st.slider("Relative humidity (%)", 15, 100, 40)
        with c11: wind = st.slider("Wind (km/h)", 0.4, 9.4, 4.0)
        with c12: rain = st.slider("Rain (mm)", 0.0, 6.4, 0.0)

        submitted = st.form_submit_button("Run prediction")

    if submitted:
        row = pd.DataFrame([{
            "X": x_coord, "Y": y_coord, "month": month, "day": day,
            "FFMC": ffmc, "DMC": dmc, "DC": dc, "ISI": isi,
            "temp": temp, "RH": rh, "wind": wind, "rain": rain,
        }])

        model_probs = bk.predict_one(pipelines, row)
        blended = float(np.mean(list(model_probs.values())))
        risk_label = "High Risk" if blended >= 0.5 else "Low Risk"
        cluster_id, cluster_label = bk.nearest_cluster(kmeans, cluster_scaler, risk_labels, row)

        st.markdown("###")
        gauge_color = COLORS["deep_red"] if blended >= 0.5 else COLORS["safe"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=blended * 100,
            number={"suffix": "%", "font": {"color": COLORS["text"], "family": "JetBrains Mono", "size": 40}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": gauge_color}, "bgcolor": COLORS["bg_card_alt"],
                   "borderwidth": 0,
                   "steps": [{"range": [0, 35], "color": "rgba(94,122,114,0.35)"},
                             {"range": [35, 65], "color": "rgba(255,186,8,0.25)"},
                             {"range": [65, 100], "color": "rgba(193,18,31,0.30)"}]},
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text_dim"]),
                           height=280, margin=dict(l=20, r=20, t=10, b=10))

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(fig, width="stretch")
            st.markdown(
                f"<div style='text-align:center;font-family:Oswald,sans-serif;font-size:1.3rem;"
                f"color:{gauge_color};text-transform:uppercase;'>{risk_label}</div>",
                unsafe_allow_html=True,
            )
        with g2:
            st.markdown("**Per-model probability of significant burn damage**")
            for name, p in model_probs.items():
                st.progress(p, text=f"{name}: {p*100:.1f}%")
            st.markdown("###")
            st.markdown(
                f"""
                <div class="ember-card">
                <span class="ember-eyebrow">NEAREST CONDITION CLUSTER</span><br>
                <span style="font-family:'Oswald',sans-serif;font-size:1.05rem;
                color:{risk_color(cluster_label)};">{cluster_label}</span><br>
                <span style="font-size:0.82rem;">These inputs most resemble historical fires in
                cluster #{cluster_id}, where {cluster_profile.loc[cluster_id,'fire_risk']*100:.0f}%
                caused measurable burn damage.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ========================================================================
# PAGE: RISK CLUSTERS
# ========================================================================
elif page == "Risk Clusters":
    st.markdown('<div class="ember-eyebrow">UNSUPERVISED · K-MEANS</div>', unsafe_allow_html=True)
    st.title("Risk Clusters")
    st.markdown("K-Means groups historical fires by environmental fingerprint alone — no labels involved.")

    df_clustered = bundle["df_clustered"]
    df_clustered["cluster_label"] = df_clustered["cluster"].map(risk_labels)

    c1, c2, c3 = st.columns(3)
    for col, label in zip((c1, c2, c3), ["Low Risk", "Moderate Risk", "High Risk"]):
        with col:
            sub = cluster_profile[cluster_profile["risk_label"] == label]
            if len(sub):
                row = sub.iloc[0]
                st.markdown(
                    f"""
                    <div class="ember-card" style="border-left:4px solid {risk_color(label)};">
                    <span class="ember-eyebrow">{label.upper()}</span><br>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;color:{risk_color(label)};">
                    {row['fire_risk']*100:.0f}%</span><span style="font-size:0.8rem;"> burn-damage rate</span><br>
                    <span style="font-size:0.78rem;color:{COLORS['smoke']};">avg temp {row['temp']:.1f}°C ·
                    avg wind {row['wind']:.1f} km/h · avg FFMC {row['FFMC']:.1f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("###")
    numeric_cols = ["X","Y","FFMC","DMC","DC","ISI","temp","RH","wind","rain"]
    c4, c5 = st.columns(2)
    with c4: x_axis = st.selectbox("X axis", numeric_cols, index=numeric_cols.index("DMC"))
    with c5: y_axis = st.selectbox("Y axis", numeric_cols, index=numeric_cols.index("temp"))

    color_map = {label: risk_color(label) for label in risk_labels.values()}
    fig = px.scatter(df_clustered, x=x_axis, y=y_axis, color="cluster_label", color_discrete_map=color_map,
                      hover_data=["FFMC","DMC","DC","temp","wind"], opacity=0.8)
    fig.update_layout(**PLOTLY_TEMPLATE, height=480)
    st.plotly_chart(fig, width="stretch")
    st.caption("Clusters fit only on weather/FWI features, then labelled by their observed burn-damage rate.")
