"""
app.py — Proactive Retention Engine 2.0
Enterprise Edge-Deployment Dashboard
Glassmorphism UI · MoE Inference · LLM Strategy · Hardware Monitor
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.graph_objects as go
import psutil
import numpy as np
import time

# ──────────────────────────────────────────────────────────
# ENSURE DARK-MODE CONFIG EXISTS
# ──────────────────────────────────────────────────────────
_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.streamlit')
_config_path = os.path.join(_config_dir, 'config.toml')
if not os.path.exists(_config_path):
    os.makedirs(_config_dir, exist_ok=True)
    with open(_config_path, 'w') as f:
        f.write(
            '[theme]\nprimaryColor = "#818cf8"\nbackgroundColor = "#0a0a1a"\n'
            'secondaryBackgroundColor = "#111827"\ntextColor = "#e2e8f0"\n'
            'font = "sans serif"\n\n[server]\nheadless = true\n\n'
            '[browser]\ngatherUsageStats = false\n'
        )

from inference import ensemble_predict, generate_retention_strategy
from model_training import GGUF_FILE, _download_gguf
from preprocessing import ARTIFACTS_DIR

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRE — Edge Deployment Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# ENTERPRISE CSS — Glassmorphism + Glow + Animations
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp { background: #0a0a1a; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
        border-right: 1px solid rgba(99,102,241,0.15);
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label {
        color: #c5c9d6 !important;
    }

    /* ── Animations ── */
    @keyframes subtleGlow {
        0%, 100% { box-shadow: 0 0 15px rgba(99,102,241,0.15); }
        50%      { box-shadow: 0 0 25px rgba(99,102,241,0.3); }
    }
    @keyframes borderPulse {
        0%, 100% { border-color: rgba(99,102,241,0.25); }
        50%      { border-color: rgba(99,102,241,0.55); }
    }
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeSlideUp 0.5s ease-out forwards;
    }

    /* ── Header Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.8rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: subtleGlow 4s ease-in-out infinite;
    }
    .hero-banner::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 30% 20%, rgba(99,102,241,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner h1 {
        color: #fff; font-size: 2.1rem; font-weight: 800; margin: 0 0 0.3rem 0;
        letter-spacing: -0.02em;
    }
    .hero-banner .sub {
        color: #94a3b8; font-size: 0.9rem; font-weight: 400;
        letter-spacing: 0.04em; text-transform: uppercase;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.35);
        color: #a5b4fc; font-size: 0.7rem; font-weight: 600;
        padding: 3px 10px; border-radius: 20px;
        margin-top: 0.6rem; letter-spacing: 0.06em; text-transform: uppercase;
    }

    /* ── Glassmorphism Cards ── */
    .glass-card {
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        animation: fadeSlideUp 0.5s ease-out forwards,
                   borderPulse 5s ease-in-out infinite;
    }
    .glass-card h3 {
        color: #64748b; font-size: 0.72rem; font-weight: 600; margin: 0 0 0.5rem 0;
        text-transform: uppercase; letter-spacing: 0.1em;
    }
    .glass-card .val {
        font-size: 2.4rem; font-weight: 800; margin: 0; line-height: 1;
    }
    .glass-card .val-sub {
        font-size: 0.8rem; color: #64748b; margin-top: 0.4rem;
    }

    /* Risk colors */
    .risk-high { color: #f87171; text-shadow: 0 0 20px rgba(248,113,113,0.3); }
    .risk-low  { color: #34d399; text-shadow: 0 0 20px rgba(52,211,153,0.3); }

    /* ── Hardware Glow Cards ── */
    .hw-glow {
        background: rgba(17, 24, 39, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        animation: subtleGlow 6s ease-in-out infinite;
    }
    .hw-glow .hw-label {
        color: #475569; font-size: 0.65rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 0.3rem 0;
    }
    .hw-glow .hw-value {
        color: #e2e8f0; font-size: 1.5rem; font-weight: 700; margin: 0;
    }
    .hw-glow .hw-detail {
        color: #475569; font-size: 0.65rem; margin: 0.2rem 0 0 0;
    }

    /* ── Strategy Card ── */
    .strategy-card {
        background: linear-gradient(135deg, rgba(16,42,76,0.7) 0%, rgba(30,64,100,0.5) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 16px;
        padding: 1.5rem;
        animation: fadeSlideUp 0.5s ease-out forwards;
    }
    .strategy-card .strat-badge {
        display: inline-block;
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.3);
        color: #93c5fd; font-size: 0.65rem; font-weight: 700;
        padding: 3px 10px; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin-bottom: 0.7rem;
    }
    .strategy-card .strat-text {
        color: #e2e8f0; font-size: 1.05rem; font-weight: 400;
        line-height: 1.65; margin: 0;
    }

    /* ── Expert Row ── */
    .expert-row {
        background: rgba(17, 24, 39, 0.4);
        border: 1px solid rgba(55, 65, 81, 0.5);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        display: flex; align-items: center; justify-content: space-between;
    }
    .expert-row .exp-name {
        color: #e2e8f0; font-size: 0.85rem; font-weight: 600; min-width: 90px;
    }
    .expert-row .exp-pred {
        font-size: 0.85rem; font-weight: 700; min-width: 55px; text-align: right;
    }
    .expert-row .exp-weight {
        color: #64748b; font-size: 0.75rem; min-width: 80px; text-align: right;
    }
    .expert-bar {
        flex: 1; margin: 0 12px; height: 6px; border-radius: 3px;
        background: rgba(55, 65, 81, 0.6); overflow: hidden;
    }
    .expert-bar-fill { height: 100%; border-radius: 3px; transition: width 0.8s ease; }

    /* ── Section Titles ── */
    .section-title {
        color: #e2e8f0; font-size: 1.05rem; font-weight: 700;
        margin: 0 0 0.8rem 0; display: flex; align-items: center; gap: 8px;
    }
    .section-title .icon { font-size: 1.1rem; }

    /* ── LLM Status ── */
    .llm-status {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.3);
        color: #34d399; font-size: 0.7rem; font-weight: 600;
        padding: 3px 10px; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.06em;
    }
    .llm-status.mock {
        background: rgba(245,158,11,0.1);
        border-color: rgba(245,158,11,0.3);
        color: #fbbf24;
    }
    .llm-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: currentColor; display: inline-block;
    }

    /* ── Landing Cards ── */
    .arch-card {
        background: rgba(17, 24, 39, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 16px;
        padding: 1.5rem;
    }
    .arch-card h3 {
        color: #e2e8f0; font-size: 1rem; font-weight: 700;
        margin: 0 0 0.8rem 0;
    }
    .arch-card p, .arch-card li {
        color: #94a3b8; font-size: 0.88rem; line-height: 1.6;
    }

    /* Hide default Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background-color: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# HEADER BANNER
# ──────────────────────────────────────────────────────────
gguf_exists = os.path.exists(os.path.join(ARTIFACTS_DIR, GGUF_FILE))
llm_badge = (
    '<span class="llm-status"><span class="llm-dot"></span> Phi-3 4-bit Active</span>'
    if gguf_exists else
    '<span class="llm-status mock"><span class="llm-dot"></span> Mock LLM — Plug &amp; Play Ready</span>'
)

st.markdown(f"""
<div class="hero-banner">
    <h1>🧠 Proactive Retention Engine 2.0</h1>
    <p class="sub">Resource-Aware Soft Mixture of Experts · Edge Deployment</p>
    <div class="hero-badge">4 Experts · Neural Router · Generative LLM</div>
    <div style="margin-top:0.6rem;">{llm_badge}</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# SIDEBAR — Customer Profile (grouped with expanders)
# ──────────────────────────────────────────────────────────
st.sidebar.markdown("### 📋 Customer Profile")

with st.sidebar.expander("👤 Demographics", expanded=False):
    gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
    SeniorCitizen = st.selectbox("Senior Citizen", ["0", "1"], key="senior")
    Partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
    Dependents = st.selectbox("Dependents", ["Yes", "No"], key="deps")

with st.sidebar.expander("📑 Account & Billing", expanded=True):
    tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ], key="payment")

with st.sidebar.expander("📡 Services", expanded=False):
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], key="multi")
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="security")
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="backup")
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="device")
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="tech")
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="tv")
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="movies")

with st.sidebar.expander("💰 Charges", expanded=True):
    MonthlyCharges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=5.0, key="monthly"
    )
    TotalCharges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0, step=50.0, key="total"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Support Notes")
support_notes = st.sidebar.text_area(
    "Latest customer interaction:",
    value="Internet has been dropping frequently and support was unresponsive.",
    height=100, key="notes",
)

st.sidebar.markdown("")
predict_btn = st.sidebar.button(
    "⚡ Run MoE Analysis", use_container_width=True, type="primary"
)

# ── Optional: Download Real LLM ──
if not gguf_exists:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Local LLM Engine")
    st.sidebar.caption(
        "Currently using **mock fallback**. Click below to download "
        "Phi-3 Mini 4-bit (~2.4 GB) for real LLM inference."
    )
    if st.sidebar.button("📥 Download Phi-3 GGUF", use_container_width=True):
        with st.sidebar:
            with st.spinner("⬇️ Downloading Phi-3 Mini (~2.4 GB)..."):
                result_path = _download_gguf()
            if result_path:
                st.success("✅ Download complete! Restart the app to use real LLM.")
            else:
                st.error("❌ Download failed. Check your connection.")

# Build full user_data dict
user_data = {
    'tenure': tenure, 'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges,
    'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Partner': Partner,
    'Dependents': Dependents, 'PhoneService': PhoneService, 'MultipleLines': MultipleLines,
    'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup, 'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport, 'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies,
    'Contract': Contract, 'PaperlessBilling': PaperlessBilling, 'PaymentMethod': PaymentMethod,
}

# ──────────────────────────────────────────────────────────
# MAIN PANEL
# ──────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("⏳ Running Soft MoE inference pipeline..."):
        result = ensemble_predict(user_data, support_notes)
        strategy = generate_retention_strategy(support_notes)

    weights = result['router_weights']
    preds = result['expert_preds']
    ensemble_prob = result['ensemble_prob']
    churn_label = result['churn_label']
    risk_cls = 'risk-high' if ensemble_prob >= 0.5 else 'risk-low'
    risk_emoji = '🔴' if ensemble_prob >= 0.5 else '🟢'

    # ════════ ROW 1 — Key Metrics ════════
    c1, c2, c3 = st.columns([1.3, 1, 1])

    with c1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <h3>Ensemble Churn Probability</h3>
            <p class="val {risk_cls}">{ensemble_prob:.1%}</p>
            <p class="val-sub">{risk_emoji} {churn_label}</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        cpu = psutil.cpu_percent(interval=0.3)
        cpu_color = '#f87171' if cpu > 80 else '#fbbf24' if cpu > 50 else '#34d399'
        st.markdown(f"""
        <div class="hw-glow">
            <p class="hw-label">CPU Utilization</p>
            <p class="hw-value" style="color:{cpu_color};">{cpu:.1f}%</p>
            <p class="hw-detail">{psutil.cpu_count(logical=False)}C / {psutil.cpu_count(logical=True)}T</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        mem = psutil.virtual_memory()
        ram_gb = mem.used / (1024**3)
        ram_total = mem.total / (1024**3)
        ram_pct = mem.percent
        ram_color = '#f87171' if ram_pct > 85 else '#fbbf24' if ram_pct > 65 else '#34d399'
        st.markdown(f"""
        <div class="hw-glow">
            <p class="hw-label">Memory (4-bit Quantized)</p>
            <p class="hw-value" style="color:{ram_color};">{ram_gb:.1f} / {ram_total:.1f} GB</p>
            <p class="hw-detail">{ram_pct:.0f}% utilized</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)

    # ════════ ROW 2 — Pie + Strategy ════════
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<p class="section-title"><span class="icon">🎯</span> Algorithmic Mix — Router Weights</p>', unsafe_allow_html=True)

        expert_names = ['XGBoost', 'SVM', 'Deep ANN', 'Local LLM']
        colors = ['#818cf8', '#fbbf24', '#34d399', '#f87171']

        fig = go.Figure(data=[go.Pie(
            labels=expert_names,
            values=weights,
            marker=dict(
                colors=colors,
                line=dict(color='#0a0a1a', width=3),
            ),
            textinfo='label+percent',
            textfont=dict(size=12, color='#e2e8f0', family='Inter'),
            hovertemplate=(
                "<b style='font-size:14px;'>%{label}</b><br>"
                "Weight: %{value:.4f}<br>"
                "Share: %{percent}<extra></extra>"
            ),
            hole=0.55,
            pull=[0.03, 0.03, 0.03, 0.03],
        )])

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Inter'),
            showlegend=True,
            legend=dict(
                orientation='h', yanchor='bottom', y=-0.12,
                xanchor='center', x=0.5,
                font=dict(size=11, color='#94a3b8'),
                bgcolor='rgba(0,0,0,0)',
            ),
            margin=dict(t=5, b=25, l=5, r=5),
            height=330,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<p class="section-title"><span class="icon">🛡️</span> AI Retention Strategy</p>', unsafe_allow_html=True)
        
        # Clean up LLM output formatting so it doesn't break the HTML/CSS box
        import re
        # 1. Remove special model tags like <|assistant|>
        safe_strategy = re.sub(r'<\|.*?\|>', '', strategy)
        # 2. Remove any internal "Next Best Action:" or "Response:" headers
        safe_strategy = re.sub(r'(?i)(next best action:|response:)\s*', '', safe_strategy)
        # 3. Strip Markdown headers and bolding
        safe_strategy = re.sub(r'(?m)^#+\s*', '', safe_strategy)
        safe_strategy = safe_strategy.replace('**', '')
        # 4. If it started hallucinating a second separate response, chop it off
        safe_strategy = safe_strategy.split('===')[0].strip()
        safe_strategy = safe_strategy.split('---')[0].strip()
        # 5. Finally apply HTML linebreaks
        safe_strategy = safe_strategy.replace('\n', '<br>').strip()

        st.markdown(f"""
        <div class="strategy-card">
            <span class="strat-badge">💡 Next Best Action</span>
            <p class="strat-text">{safe_strategy}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-title"><span class="icon">📊</span> Individual Expert Predictions</p>', unsafe_allow_html=True)

        expert_colors = dict(zip(expert_names, colors))
        for name, pred, weight in zip(expert_names, preds, weights):
            pred_pct = pred * 100
            bar_w = min(pred * 100, 100)
            c = expert_colors[name]
            pred_cls = 'risk-high' if pred >= 0.5 else 'risk-low'
            st.markdown(f"""
            <div class="expert-row">
                <span class="exp-name" style="color:{c};">● {name}</span>
                <div class="expert-bar">
                    <div class="expert-bar-fill" style="width:{bar_w}%; background:{c};"></div>
                </div>
                <span class="exp-pred {pred_cls}">{pred_pct:.1f}%</span>
                <span class="exp-weight">w={weight:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

    # ════════ ROW 3 — Hardware Monitor ════════
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-title"><span class="icon">⚙️</span> Hardware Monitor — Edge Efficiency</p>', unsafe_allow_html=True)

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown(f"""
        <div class="hw-glow">
            <p class="hw-label">CPU Cores</p>
            <p class="hw-value">{psutil.cpu_count(logical=False)} / {psutil.cpu_count(logical=True)}</p>
            <p class="hw-detail">Physical / Logical</p>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div class="hw-glow">
            <p class="hw-label">CPU Load</p>
            <p class="hw-value" style="color:{cpu_color};">{cpu:.1f}%</p>
            <p class="hw-detail">Real-time</p>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown(f"""
        <div class="hw-glow">
            <p class="hw-label">RAM Used</p>
            <p class="hw-value" style="color:{ram_color};">{ram_gb:.2f} GB</p>
            <p class="hw-detail">of {ram_total:.1f} GB</p>
        </div>
        """, unsafe_allow_html=True)
    with h4:
        disk = psutil.disk_usage('/')
        disk_color = '#f87171' if disk.percent > 90 else '#34d399'
        st.markdown(f"""
        <div class="hw-glow">
            <p class="hw-label">Disk Usage</p>
            <p class="hw-value" style="color:{disk_color};">{disk.percent:.0f}%</p>
            <p class="hw-detail">{disk.used / (1024**3):.0f} / {disk.total / (1024**3):.0f} GB</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # ════════ LANDING STATE ════════
    st.markdown("""
    <div class="glass-card" style="text-align:center; margin-bottom:1.5rem;">
        <p style="color:#94a3b8; font-size:0.95rem; margin:0;">
            👈 Configure the <strong style="color:#e2e8f0;">Customer Profile</strong> in the sidebar,
            then click <strong style="color:#818cf8;">⚡ Run MoE Analysis</strong>
            to execute the Mixture of Experts inference pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="arch-card">
            <h3>🏗️ Architecture</h3>
            <p>This dashboard runs a <strong>Soft Mixture of Experts</strong> architecture:</p>
            <ol>
                <li><strong>4 Expert Models</strong> produce independent churn forecasts</li>
                <li><strong>Neural Router</strong> assigns per-customer trust weights</li>
                <li><strong>Weighted Ensemble</strong> fuses predictions into a final score</li>
                <li><strong>Generative LLM</strong> suggests a personalized retention strategy</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="arch-card">
            <h3>🤖 Expert Pool</h3>
            <table style="width:100%; color:#94a3b8; font-size:0.85rem; border-collapse:collapse;">
                <tr style="border-bottom:1px solid rgba(55,65,81,0.5);">
                    <td style="padding:6px 0;">🌲 <strong style="color:#818cf8;">XGBoost</strong></td>
                    <td>Gradient Boosting</td><td>19 Tabular</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(55,65,81,0.5);">
                    <td style="padding:6px 0;">📐 <strong style="color:#fbbf24;">SVM</strong></td>
                    <td>Support Vectors</td><td>19 Tabular</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(55,65,81,0.5);">
                    <td style="padding:6px 0;">🧠 <strong style="color:#34d399;">Deep ANN</strong></td>
                    <td>Keras DNN</td><td>19 Tabular</td>
                </tr>
                <tr>
                    <td style="padding:6px 0;">💬 <strong style="color:#f87171;">Local LLM</strong></td>
                    <td>Phi-3 4-bit GGUF</td><td>Text</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
