import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="ChurnIQ — Insights",
    page_icon="💡",
    layout="wide"
)

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

@st.cache_resource
def load_assets():
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    assets = {
        "model"    : joblib.load(f"{base}/models/final_model.pkl"),
        "features" : joblib.load(f"{base}/models/feature_names.pkl"),
    }
    # Optional SHAP assets
    shap_path = f"{base}/outputs/shap_feature_importance.csv"
    if os.path.exists(shap_path):
        assets["shap_df"] = pd.read_csv(shap_path)
    seg_path = f"{base}/data/segmented_churn.csv"
    if os.path.exists(seg_path):
        assets["seg_df"] = pd.read_csv(seg_path)
    orig_path = f"{base}/data/cleaned_churn.csv"
    if os.path.exists(orig_path):
        assets["orig_df"] = pd.read_csv(orig_path)
    return assets

assets = load_assets()

# ── PAGE HEADER ──
st.markdown("""
<div style="margin-bottom:2rem">
    <div style="color:#4F7FFF;font-size:0.75rem;font-weight:600;
                letter-spacing:0.1em;text-transform:uppercase;
                margin-bottom:8px;margin-top:50px">Explainable AI</div>
    <h1 style="font-size:1.8rem;font-weight:700;color:#F0F4FF;
               margin:0;letter-spacing:-0.01em">Model Insights</h1>
    <p style="color:#8B97B8;font-size:0.9rem;margin-top:6px">
        Feature importance, churn drivers, and customer segment analysis.
    </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3, tab4 = st.tabs([
    "  📊  Feature Importance  ",
    "  🧩  Segment Profiles  ",
    "  📈  EDA Charts  ",
    "  🤖  Model Performance  ",
])

# ════════════════════════════════════════════════
# TAB 1 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════
with tab1:
    st.markdown("<div style='margin:1rem 0'>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div style="color:#F0F4FF;font-size:0.9rem;font-weight:600;
                    margin-bottom:1rem">SHAP Feature Importance</div>
        """, unsafe_allow_html=True)

        if "shap_df" in assets:
            shap_df = assets["shap_df"].head(15)

            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#1C2333')
            ax.set_facecolor('#1C2333')

            colors = plt.cm.RdYlGn_r(
                np.linspace(0.1, 0.9, len(shap_df))
            )
            bars = ax.barh(
                shap_df['Feature'][::-1],
                shap_df['SHAP_Score'][::-1],
                color=colors[::-1],
                edgecolor='none',
                height=0.65
            )
            ax.set_xlabel('Mean |SHAP Value|', color='#8B97B8', fontsize=9)
            ax.tick_params(colors='#8B97B8', labelsize=8)
            ax.spines[:].set_color('#2A3350')
            ax.xaxis.grid(True, color='#2A3350', linewidth=0.5)
            ax.set_axisbelow(True)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.0003, bar.get_y() + bar.get_height()/2,
                        f'{w:.4f}', va='center', ha='left',
                        color='#8B97B8', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("Run Phase 8 notebook to generate SHAP values.")

    with col_right:
        st.markdown("""
        <div style="color:#F0F4FF;font-size:0.9rem;font-weight:600;
                    margin-bottom:1rem">Top Churn Drivers</div>
        """, unsafe_allow_html=True)

        drivers = [
            ("IsLongTermContract",      "0.0721", "No long-term contract = very high churn risk",       "#FF4B6E"),
            ("ChargesPerTenure",        "0.0552", "High early charges signal low perceived value",      "#FF8C42"),
            ("Contract_Month-to-month", "0.0527", "Month-to-month = easiest to leave",                 "#FF8C42"),
            ("InternetService_Fiber",   "0.0446", "Fiber users churn despite premium pricing",         "#FFD166"),
            ("Contract_Two year",       "0.0318", "Two-year contract strongly reduces churn risk",     "#06D6A0"),
        ]

        for feat, score, insight, color in drivers:
            st.markdown(f"""
            <div style="background:#161B27;border:1px solid #2A3350;
                        border-left:3px solid {color};border-radius:8px;
                        padding:0.9rem;margin-bottom:0.6rem">
                <div style="display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:4px">
                    <span style="color:#F0F4FF;font-size:0.78rem;
                                 font-weight:600;font-family:'JetBrains Mono',monospace">{feat}</span>
                    <span style="color:{color};font-size:0.78rem;
                                 font-weight:700">{score}</span>
                </div>
                <div style="color:#8B97B8;font-size:0.74rem;
                            line-height:1.4">{insight}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2A3350;margin:1.5rem 0'>", unsafe_allow_html=True)

    # RF Native Importance
    st.markdown("""
    <div style="color:#F0F4FF;font-size:0.9rem;font-weight:600;
                margin-bottom:1rem">Random Forest — Native Feature Importance</div>
    """, unsafe_allow_html=True)

    importances = assets["model"].feature_importances_
    feat_names  = assets["features"]
    imp_df = pd.DataFrame({
        'Feature': feat_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    fig2.patch.set_facecolor('#1C2333')
    ax2.set_facecolor('#1C2333')

    ax2.bar(imp_df['Feature'], imp_df['Importance'],
            color='#4F7FFF', edgecolor='none', alpha=0.85)
    ax2.tick_params(axis='x', rotation=45, colors='#8B97B8', labelsize=8)
    ax2.tick_params(axis='y', colors='#8B97B8', labelsize=8)
    ax2.spines[:].set_color('#2A3350')
    ax2.yaxis.grid(True, color='#2A3350', linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_ylabel('Importance', color='#8B97B8', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

# ════════════════════════════════════════════════
# TAB 2 — SEGMENT PROFILES
# ════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='margin:1rem 0'>", unsafe_allow_html=True)

    segments = [
        {
            "name"    : "Critical Churn-Risk",
            "churn"   : "54.7%",
            "tenure"  : "21 months",
            "monthly" : "$87",
            "addons"  : "1.9",
            "longterm": "1%",
            "fiber"   : "100%",
            "color"   : "#FF4B6E",
            "bg"      : "rgba(255,75,110,0.06)",
            "priority": "🔴 Critical",
            "actions" : [
                "Immediate retention call within 24 hours",
                "Offer 15–20% discount for 3–6 months",
                "Contract upgrade with 2 free months incentive",
                "Bundle OnlineSecurity + TechSupport at 50% off",
                "Fiber service review — speed upgrade or bill reduction",
            ]
        },
        {
            "name"    : "Vulnerable High-Risk",
            "churn"   : "28.5%",
            "tenure"  : "13 months",
            "monthly" : "$41",
            "addons"  : "1.1",
            "longterm": "3%",
            "fiber"   : "0%",
            "color"   : "#FF8C42",
            "bg"      : "rgba(255,140,66,0.06)",
            "priority": "🟠 High",
            "actions" : [
                "Early engagement: personalized welcome email",
                "First add-on service free for 2 months",
                "Promote annual contract with savings highlight",
                "Upsell enhanced plan with streaming bundle",
            ]
        },
        {
            "name"    : "Stable Mid-Risk",
            "churn"   : "9.2%",
            "tenure"  : "56 months",
            "monthly" : "$84",
            "addons"  : "4.1",
            "longterm": "96%",
            "fiber"   : "46%",
            "color"   : "#FFD166",
            "bg"      : "rgba(255,209,102,0.06)",
            "priority": "🟡 Medium",
            "actions" : [
                "Quarterly loyalty reward — bonus data or credit",
                "Cross-sell DeviceProtection or StreamingTV",
                "Proactive fiber quality check",
                "Promote auto-pay enrollment for discount",
            ]
        },
        {
            "name"    : "Loyal Low-Risk",
            "churn"   : "2.0%",
            "tenure"  : "42 months",
            "monthly" : "$25",
            "addons"  : "0.2",
            "longterm": "100%",
            "fiber"   : "2%",
            "color"   : "#06D6A0",
            "bg"      : "rgba(6,214,160,0.06)",
            "priority": "🟢 Low",
            "actions" : [
                "Annual loyalty thank-you reward",
                "Premium bundle offer at 30% off",
                "Referral program invitation",
            ]
        },
    ]

    for seg in segments:
        st.markdown(f"""
        <div style="background:{seg['bg']};border:1px solid {seg['color']};
                    border-radius:12px;padding:1.5rem;margin-bottom:1rem">
            <div style="display:flex;justify-content:space-between;
                        align-items:flex-start;flex-wrap:wrap;gap:1rem">

                <div style="flex:1;min-width:200px">
                    <div style="color:{seg['color']};font-size:0.72rem;
                                font-weight:600;letter-spacing:0.08em;
                                text-transform:uppercase;margin-bottom:6px">
                        {seg['priority']}
                    </div>
                    <div style="color:#F0F4FF;font-size:1rem;
                                font-weight:700;margin-bottom:12px">{seg['name']}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                        <div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:8px">
                            <div style="color:#556080;font-size:0.68rem;
                                        text-transform:uppercase;letter-spacing:0.06em">Churn Rate</div>
                            <div style="color:{seg['color']};font-size:1.1rem;
                                        font-weight:700">{seg['churn']}</div>
                        </div>
                        <div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:8px">
                            <div style="color:#556080;font-size:0.68rem;
                                        text-transform:uppercase;letter-spacing:0.06em">Avg Tenure</div>
                            <div style="color:#F0F4FF;font-size:1.1rem;
                                        font-weight:700">{seg['tenure']}</div>
                        </div>
                        <div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:8px">
                            <div style="color:#556080;font-size:0.68rem;
                                        text-transform:uppercase;letter-spacing:0.06em">Monthly</div>
                            <div style="color:#F0F4FF;font-size:1.1rem;
                                        font-weight:700">{seg['monthly']}</div>
                        </div>
                        <div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:8px">
                            <div style="color:#556080;font-size:0.68rem;
                                        text-transform:uppercase;letter-spacing:0.06em">Avg Add-ons</div>
                            <div style="color:#F0F4FF;font-size:1.1rem;
                                        font-weight:700">{seg['addons']}</div>
                        </div>
                    </div>
                </div>

                <div style="flex:2;min-width:280px">
                    <div style="color:#8B97B8;font-size:0.75rem;
                                font-weight:600;margin-bottom:8px;
                                text-transform:uppercase;letter-spacing:0.06em">
                        Retention Actions
                    </div>
                    {''.join([f"""
                    <div style="display:flex;align-items:flex-start;
                                gap:8px;margin-bottom:6px">
                        <div style="width:5px;height:5px;border-radius:50%;
                                    background:{seg['color']};
                                    margin-top:6px;flex-shrink:0"></div>
                        <div style="color:#F0F4FF;font-size:0.82rem;
                                    line-height:1.4">{action}</div>
                    </div>""" for action in seg['actions']])}
                </div>

            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 3 — EDA CHARTS
# ════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='margin:1rem 0'>", unsafe_allow_html=True)

    if "orig_df" in assets:
        df = assets["orig_df"].copy()
        if df['Churn'].dtype == object:
            df['Churn'] = df['Churn'].map({'Yes':1,'No':0})

        plt_cfg = {
            'facecolor': '#1C2333',
            'text_color': '#8B97B8',
            'grid_color': '#2A3350',
            'bar_colors': ['#4F7FFF','#FF4B6E'],
        }

        def style_ax(ax):
            ax.set_facecolor(plt_cfg['facecolor'])
            ax.tick_params(colors=plt_cfg['text_color'], labelsize=8)
            ax.spines[:].set_color(plt_cfg['grid_color'])
            ax.yaxis.grid(True, color=plt_cfg['grid_color'], linewidth=0.5)
            ax.set_axisbelow(True)
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), color=plt_cfg['text_color'], fontsize=9)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), color=plt_cfg['text_color'], fontsize=9)

        # Row 1
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor('#161B27')

        # Churn by Contract
        contract_churn = df.groupby('Contract')['Churn'].mean() * 100
        axes[0].bar(contract_churn.index, contract_churn.values,
                    color=['#4F7FFF','#FF8C42','#FF4B6E'], edgecolor='none')
        axes[0].set_title('Churn Rate by Contract', color='#F0F4FF', fontsize=10, fontweight='600')
        axes[0].set_ylabel('Churn Rate (%)')
        axes[0].tick_params(axis='x', rotation=15)
        style_ax(axes[0])

        # Churn by Internet
        inet_churn = df.groupby('InternetService')['Churn'].mean() * 100
        axes[1].bar(inet_churn.index, inet_churn.values,
                    color=['#4F7FFF','#FF4B6E','#06D6A0'], edgecolor='none')
        axes[1].set_title('Churn Rate by Internet Service', color='#F0F4FF', fontsize=10, fontweight='600')
        axes[1].set_ylabel('Churn Rate (%)')
        style_ax(axes[1])

        # Tenure histogram
        for val, color in [(0,'#06D6A0'),(1,'#FF4B6E')]:
            sub = df[df['Churn']==val]['tenure']
            axes[2].hist(sub, bins=25, alpha=0.65, color=color,
                         label='Retained' if val==0 else 'Churned', edgecolor='none')
        axes[2].set_title('Tenure Distribution', color='#F0F4FF', fontsize=10, fontweight='600')
        axes[2].set_xlabel('Tenure (months)')
        axes[2].set_ylabel('Count')
        axes[2].legend(fontsize=8, facecolor='#1C2333',
                       labelcolor='#8B97B8', edgecolor='#2A3350')
        style_ax(axes[2])

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Row 2
        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
        fig2.patch.set_facecolor('#161B27')

        # Monthly charges
        for val, color in [(0,'#06D6A0'),(1,'#FF4B6E')]:
            sub = df[df['Churn']==val]['MonthlyCharges']
            axes2[0].hist(sub, bins=25, alpha=0.65, color=color,
                          label='Retained' if val==0 else 'Churned', edgecolor='none')
        axes2[0].set_title('Monthly Charges Distribution', color='#F0F4FF', fontsize=10, fontweight='600')
        axes2[0].set_xlabel('Monthly Charges ($)')
        axes2[0].legend(fontsize=8, facecolor='#1C2333',
                        labelcolor='#8B97B8', edgecolor='#2A3350')
        style_ax(axes2[0])

        # Payment method
        pay_churn = df.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=False) * 100
        short_labels = [p.replace(' (automatic)','*').replace(' check','') for p in pay_churn.index]
        axes2[1].barh(short_labels, pay_churn.values,
                      color='#FF8C42', edgecolor='none')
        axes2[1].set_title('Churn by Payment Method', color='#F0F4FF', fontsize=10, fontweight='600')
        axes2[1].set_xlabel('Churn Rate (%)')
        style_ax(axes2[1])

        # Senior citizen
        senior_churn = df.groupby('SeniorCitizen')['Churn'].mean() * 100
        labels = ['Non-Senior','Senior'] if 0 in senior_churn.index else senior_churn.index
        axes2[2].bar(labels, senior_churn.values,
                     color=['#4F7FFF','#FF4B6E'], edgecolor='none', width=0.5)
        axes2[2].set_title('Churn by Senior Status', color='#F0F4FF', fontsize=10, fontweight='600')
        axes2[2].set_ylabel('Churn Rate (%)')
        style_ax(axes2[2])

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    else:
        st.info("Cleaned dataset not found. Ensure data/cleaned_churn.csv exists.")

# ════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════
with tab4:
    st.markdown("<div style='margin:1rem 0'>", unsafe_allow_html=True)

    # Model comparison table
    st.markdown("""
    <div style="color:#F0F4FF;font-size:0.9rem;font-weight:600;
                margin-bottom:1rem">Model Comparison</div>
    """, unsafe_allow_html=True)

    perf_data = {
        'Model'    : ['Logistic Regression','Decision Tree',
                      'Random Forest (Final)','XGBoost'],
        'Accuracy' : [0.7406, 0.7349, 0.7498, 0.7463],
        'Precision': [0.5081, 0.5009, 0.5200, 0.5166],
        'Recall'   : [0.7567, 0.7326, 0.7647, 0.7059],
        'F1'       : [0.6079, 0.5950, 0.6190, 0.5966],
        'ROC_AUC'  : [0.8241, 0.8081, 0.8305, 0.8206],
    }
    perf_df = pd.DataFrame(perf_data)

    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color:#1a3a2a;color:#06D6A0;font-weight:700'
                if v else 'color:#F0F4FF' for v in is_max]

    def highlight_model(row):
        if row['Model'] == 'Random Forest (Final)':
            return ['background-color:rgba(79,127,255,0.08);'
                    'color:#F0F4FF'] * len(row)
        return ['color:#8B97B8'] * len(row)

    styled_perf = (
        perf_df.style
        .apply(highlight_best, subset=['Accuracy','Precision','Recall','F1','ROC_AUC'])
        .apply(highlight_model, axis=1)
        .set_properties(**{'background-color':'#1C2333','border-color':'#2A3350','font-size':'13px'})
        .format({'Accuracy':'{:.4f}','Precision':'{:.4f}',
                 'Recall':'{:.4f}','F1':'{:.4f}','ROC_AUC':'{:.4f}'})
        .hide(axis='index')
    )
    st.dataframe(styled_perf, use_container_width=True)

    st.markdown("<div style='margin:1.5rem 0'>", unsafe_allow_html=True)

    # Final model metrics
    st.markdown("""
    <div style="color:#F0F4FF;font-size:0.9rem;font-weight:600;
                margin-bottom:1rem">Final Model — Random Forest</div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    final_metrics = [
        ("ROC-AUC",   "0.8327", "#4F7FFF"),
        ("Accuracy",  "75.6%",  "#4F7FFF"),
        ("Precision", "52.7%",  "#FF8C42"),
        ("Recall",    "78.1%",  "#06D6A0"),
        ("F1 Score",  "0.6293", "#FFD166"),
    ]
    for col, (label, val, color) in zip([c1,c2,c3,c4,c5], final_metrics):
        with col:
            st.markdown(f"""
            <div style="background:#1C2333;border:1px solid #2A3350;
                        border-left:3px solid {color};
                        border-radius:10px;padding:1rem;text-align:center">
                <div style="color:#8B97B8;font-size:0.7rem;text-transform:uppercase;
                            letter-spacing:0.07em;margin-bottom:6px">{label}</div>
                <div style="color:{color};font-size:1.5rem;
                            font-weight:700">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1.5rem 0'>", unsafe_allow_html=True)

    # Selection rationale
    st.markdown("""
    <div style="background:#1C2333;border:1px solid #2A3350;
                border-radius:12px;padding:1.5rem">
        <div style="color:#F0F4FF;font-size:0.9rem;font-weight:600;
                    margin-bottom:1rem">Selection Rationale</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
            <div>
                <div style="color:#4F7FFF;font-size:0.75rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.06em;
                            margin-bottom:6px">Why Random Forest?</div>
                <div style="color:#8B97B8;font-size:0.83rem;line-height:1.7">
                    • Highest ROC-AUC (0.8305) across all models<br>
                    • Best F1-score (0.6190) on imbalanced data<br>
                    • Robust to overfitting via bagging<br>
                    • Native feature importance for explainability<br>
                    • Strong recall (76%) — catches most churners
                </div>
            </div>
            <div>
                <div style="color:#06D6A0;font-size:0.75rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.06em;
                            margin-bottom:6px">Key Design Decisions</div>
                <div style="color:#8B97B8;font-size:0.83rem;line-height:1.7">
                    • SMOTE applied on training set only<br>
                    • Optimal threshold 0.51 (maximizes F1)<br>
                    • StandardScaler on continuous features<br>
                    • class_weight='balanced' for imbalance<br>
                    • 5-fold stratified cross-validation
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
