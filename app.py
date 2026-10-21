import os
import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection · AI Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS — clean, modern look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --primary: #6a1b9a;
            --primary-soft: #f3e8ff;
            --danger: #d32f2f;
            --success: #2e7d32;
            --surface: #ffffff;
            --muted: #6b7280;
        }

        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        .main {background-color: #fafafa;}

        h1, h2, h3 {color: var(--primary); font-weight: 700;}

        /* KPI cards */
        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid #ececec;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.85rem;
            color: var(--muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 700;
        }

        /* Buttons */
        .stButton > button, .stDownloadButton > button {
            background: var(--primary);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #4a1370;
            color: white;
        }

        /* Upload area */
        section[data-testid="stFileUploaderDropzone"] {
            border: 2px dashed var(--primary);
            border-radius: 14px;
            background: var(--primary-soft);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {gap: 6px;}
        .stTabs [data-baseweb="tab"] {
            background: #f1eafd;
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: var(--primary);
            color: white !important;
        }

        /* Alert tweaks */
        .stAlert {border-radius: 12px;}

        /* Hero */
        .hero {
            background: linear-gradient(135deg, #6a1b9a 0%, #9c27b0 100%);
            padding: 26px 28px;
            border-radius: 16px;
            color: white;
            margin-bottom: 20px;
        }
        .hero h1 {color: white; margin: 0 0 6px 0; font-size: 1.9rem;}
        .hero p  {color: #f3e8ff; margin: 0; font-size: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# LOAD MODEL (cached so it only loads once)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_detection_model.pkl")


@st.cache_resource(show_spinner="Loading fraud detection model…")
def load_model(path: str):
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"⚠️ Model file not found at:\n`{MODEL_PATH}`\n\n"
        "Make sure `fraud_detection_model.pkl` is in the same folder as `app.py`."
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Fraud Detection Console")
    st.caption("Upload transaction data to score each row for fraud risk.")
    st.divider()
    st.markdown("**Required columns**")
    st.code("v1 … v28, amount", language="text")
    st.caption("Extra columns (e.g. `Time`, `Class`) are ignored gracefully.")
    st.divider()
    st.markdown("**Decision threshold**")
    threshold = st.slider(
        "Flag as fraud if probability ≥",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Lower = catch more fraud (higher recall), more false alarms.",
    )
    st.caption("Tip: 0.10–0.30 catches more fraud at the cost of more alerts.")
    st.divider()
    if st.button("🧹 Clear results", use_container_width=True):
        for key in ["results_df", "results_name", "scored_at"]:
            st.session_state.pop(key, None)
        st.rerun()
    st.caption("Model loaded ✅")

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🔍 AI-Powered Credit Card Fraud Detection</h1>
        <p>Upload a CSV or Excel file of transactions (V1–V28 + Amount).
        The model scores each transaction and flags likely fraud.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
REQUIRED = [f"v{i}" for i in range(1, 29)] + ["amount"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase & trim all column names, strip a leading BOM if present."""
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]
    return df


def build_excel(df: pd.DataFrame) -> bytes:
    """Build an in-memory Excel workbook with two sheets."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
        summary = (
            df["prediction"].value_counts().rename_axis("prediction").reset_index(name="count")
        )
        summary.to_excel(writer, sheet_name="Summary", index=False)
    return buffer.getvalue()


def kpi_card(col, label, value, delta=None, delta_color="normal", help_text=None):
    with col:
        st.metric(label=label, value=value, delta=delta, delta_color=delta_color, help=help_text)


# ---------------------------------------------------------------------------
# UPLOAD & PROCESS
# ---------------------------------------------------------------------------
st.subheader("📤 Upload Transactions")
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    label_visibility="collapsed",
)

# Persist results across reruns (so the threshold slider doesn't wipe them)
if "results_df" not in st.session_state:
    st.session_state.results_df = None
    st.session_state.results_name = None
    st.session_state.scored_at = None

if uploaded_file is not None:
    try:
        # ---- Read file ----
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            raw = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            raw = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload .csv, .xlsx, or .xls.")
            st.stop()

        if raw.empty:
            st.warning("The uploaded file is empty.")
            st.stop()

        data = normalize_columns(raw)

        # ---- Column validation ----
        missing = [c for c in REQUIRED if c not in data.columns]
        if missing:
            st.error(
                "❌ Missing required columns: "
                f"`{', '.join(missing)}`\n\n"
                "Please ensure the file includes **V1–V28** and **Amount**."
            )
            with st.expander("🔎 Detected columns"):
                st.write(list(data.columns))
            st.stop()

        # ---- Preview ----
        with st.expander(f"👀 Data preview — {len(data):,} rows", expanded=False):
            st.dataframe(data.head(20), use_container_width=True)

        # ---- Predict ----
        x = data[REQUIRED].astype(float).fillna(0.0)
        with st.spinner("Scoring transactions…"):
            proba = model.predict_proba(x)[:, 1]
            preds = (proba >= threshold).astype(int)

        scored = data.copy()
        scored["prediction"] = np.where(preds == 1, "Fraud", "Legitimate")
        scored["fraud_probability_pct"] = (proba * 100).round(2)

        st.session_state.results_df = scored
        st.session_state.results_name = uploaded_file.name
        st.session_state.scored_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# RESULTS DASHBOARD
# ---------------------------------------------------------------------------
scored = st.session_state.results_df

if scored is None:
    st.info("👆 Upload a file above to begin fraud detection.")
else:
    total = len(scored)
    n_fraud = int((scored["prediction"] == "Fraud").sum())
    n_legit = total - n_fraud
    pct_fraud = (n_fraud / total * 100) if total else 0.0
    avg_prob = scored["fraud_probability_pct"].mean() if total else 0.0
    amount_fraud = (
        scored.loc[scored["prediction"] == "Fraud", "amount"].sum()
        if "amount" in scored.columns else np.nan
    )

    st.success(
        f"✅ Scored **{total:,}** transactions from "
        f"`{st.session_state.results_name}` at {st.session_state.scored_at} "
        f"(threshold = {threshold:.2f})."
    )

    # ---- KPI row ----
    st.subheader("📊 Overview")
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Transactions Scored", f"{total:,}")
    kpi_card(c2, "Fraud Flagged", f"{n_fraud:,}", delta=f"{pct_fraud:.2f}% of total",
             delta_color="inverse")
    kpi_card(c3, "Legitimate", f"{n_legit:,}")
    kpi_card(c4, "Average Fraud Probability",
             f"{avg_prob:.1f}%" if total else "—",
             help_text="Mean fraud probability across all scored rows.")

    if not np.isnan(amount_fraud):
        st.caption(f"💵 Total amount flagged as fraud: **${amount_fraud:,.2f}**")

    # ---- Tabs ----
    tab_viz, tab_table, tab_download = st.tabs(
        ["📈 Visualizations", "📋 Detailed Results", "💾 Download"]
    )

    # ============ TAB 1: VISUALIZATIONS ============
    with tab_viz:
        # Row 1: Pie + Bar
        col_a, col_b = st.columns((1, 1.3))

        with col_a:
            st.markdown("**Prediction split**")
            fig_pie = px.pie(
                names=["Legitimate", "Fraud"],
                values=[n_legit, n_fraud],
                color=["Legitimate", "Fraud"],
                color_discrete_map={"Legitimate": "#4caf50", "Fraud": "#d32f2f"},
                hole=0.55,
            )
            fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
            fig_pie.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=340,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.markdown("**Fraud probability distribution**")
            fig_hist = px.histogram(
                scored, x="fraud_probability_pct", nbins=40,
                color="prediction",
                color_discrete_map={"Legitimate": "#4caf50", "Fraud": "#d32f2f"},
                labels={"fraud_probability_pct": "Fraud probability (%)"},
            )
            fig_hist.add_vline(
                x=threshold * 100, line_dash="dash", line_color="#6a1b9a",
                annotation_text=f"Threshold {threshold:.0%}",
                annotation_position="top",
            )
            fig_hist.update_layout(
                bargap=0.05, height=340,
                margin=dict(t=10, b=10, l=10, r=10),
                legend_title_text="Prediction",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Row 2: Amount distribution (only if 'amount' present)
        if "amount" in scored.columns:
            st.markdown("**Transaction amount by prediction (log scale)**")
            plot_df = scored.copy()
            plot_df["amount"] = plot_df["amount"].clip(lower=0.01)
            fig_amt = px.box(
                plot_df, x="prediction", y="amount",
                color="prediction", log_y=True,
                color_discrete_map={"Legitimate": "#4caf50", "Fraud": "#d32f2f"},
                labels={"amount": "Amount ($, log scale)", "prediction": ""},
            )
            fig_amt.update_layout(
                showlegend=False, height=380,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_amt, use_container_width=True)

        # Row 3: Top-10 highest-risk transactions
        st.markdown("**Top 10 highest-risk transactions**")
        top = (
            scored.sort_values("fraud_probability_pct", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        display_cols = ["fraud_probability_pct"] + [
            c for c in ["amount", "prediction"] if c in top.columns
        ] + [c for c in top.columns if c not in {"fraud_probability_pct", "amount", "prediction"}][:6]
        st.dataframe(
            top[display_cols].style.background_gradient(
                subset=["fraud_probability_pct"], cmap="Reds"
            ),
            use_container_width=True,
        )

    # ============ TAB 2: DETAILED TABLE ============
    with tab_table:
        st.markdown(f"**All {total:,} scored transactions**")
        search = st.text_input("🔎 Filter (matches any column, case-insensitive)", "")

        view = scored
        if search:
            mask = scored.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            view = scored[mask]

        st.caption(f"Showing {len(view):,} of {total:,} rows")

        # Cap the rows we render (and style) so the browser stays responsive
        MAX_ROWS = 2000
        show = view.head(MAX_ROWS)
        if len(view) > MAX_ROWS:
            st.caption(
                f"⚠️ Only the first {MAX_ROWS:,} rows are rendered here. "
                "Use the Download tab to get all rows."
            )

        # Only apply the styled gradient when the table is small enough.
        if "fraud_probability_pct" in show.columns and show.size <= 100_000:
            st.dataframe(
                show.style.background_gradient(
                    subset=["fraud_probability_pct"], cmap="Reds"
                ),
                use_container_width=True,
                height=500,
            )
        else:
            # Fallback: progress bar column — looks just as good, zero cost
            st.dataframe(
                show,
                column_config={
                    "fraud_probability_pct": st.column_config.ProgressColumn(
                        "Fraud probability (%)",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.2f%%",
                    )
                },
                use_container_width=True,
                height=500,
            )

    # ============ TAB 3: DOWNLOAD ============
    with tab_download:
        st.markdown("### 💾 Export Results")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        col_x, col_y = st.columns(2)
        with col_x:
            csv_bytes = scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv_bytes,
                file_name=f"fraud_results_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_y:
            try:
                xlsx_bytes = build_excel(scored)
                st.download_button(
                    "⬇️ Download Excel (2 sheets)",
                    data=xlsx_bytes,
                    file_name=f"fraud_results_{ts}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"(Excel export unavailable: {e})")

        st.caption(
            "CSV contains all original columns plus `prediction` and "
            "`fraud_probability_pct`. The Excel file has a second 'Summary' sheet."
        )