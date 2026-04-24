import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Colour theme ──────────────────────────────────────────────────────────────
COLOURS = {'H': '#1A5C38', 'D': '#B45309', 'A': '#8B0000'}
FEATURES = ['HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'HTHG', 'HTAG']

FEATURE_LABELS = {
    'HST':  'Home Shots on Target',
    'AST':  'Away Shots on Target',
    'HC':   'Home Corners',
    'AC':   'Away Corners',
    'HY':   'Home Yellow Cards',
    'AY':   'Away Yellow Cards',
    'HR':   'Home Red Cards',
    'AR':   'Away Red Cards',
    'HTHG': 'Half Time Home Goals',
    'HTAG': 'Half Time Away Goals',
}

FEATURE_MAX = {
    'HST': 16, 'AST': 15, 'HC': 18, 'AC': 18,
    'HY': 7,   'AY': 8,   'HR': 2,  'AR': 2,
    'HTHG': 7, 'HTAG': 8,
}

# ── Train model on load ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv('EPL_combined.csv')
    df['FTR_encoded'] = df['FTR'].map({'A': 0, 'D': 1, 'H': 2})
    X = df[FEATURES]
    y = df['FTR_encoded']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

model, scaler = load_model()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='EPL Match Outcome Predictor',
    page_icon='⚽',
    layout='wide'
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#1A5C38;'>⚽ EPL Match Outcome Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#555;'>DATA 200 — Applied Statistical Analysis | "
    "Multinomial Logistic Regression</p>",
    unsafe_allow_html=True
)
st.divider()

# ── Layout: inputs left, results right ───────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap='large')

with col_left:
    st.subheader('Enter In-Match Statistics')

    st.markdown('**Shots on Target**')
    c1, c2 = st.columns(2)
    HST  = c1.number_input(FEATURE_LABELS['HST'],  0, FEATURE_MAX['HST'],  5, key='HST')
    AST  = c2.number_input(FEATURE_LABELS['AST'],  0, FEATURE_MAX['AST'],  4, key='AST')

    st.markdown('**Corners**')
    c3, c4 = st.columns(2)
    HC   = c3.number_input(FEATURE_LABELS['HC'],   0, FEATURE_MAX['HC'],   5, key='HC')
    AC   = c4.number_input(FEATURE_LABELS['AC'],   0, FEATURE_MAX['AC'],   4, key='AC')

    st.markdown('**Yellow Cards**')
    c5, c6 = st.columns(2)
    HY   = c5.number_input(FEATURE_LABELS['HY'],   0, FEATURE_MAX['HY'],   2, key='HY')
    AY   = c6.number_input(FEATURE_LABELS['AY'],   0, FEATURE_MAX['AY'],   2, key='AY')

    st.markdown('**Red Cards**')
    c7, c8 = st.columns(2)
    HR   = c7.number_input(FEATURE_LABELS['HR'],   0, FEATURE_MAX['HR'],   0, key='HR')
    AR   = c8.number_input(FEATURE_LABELS['AR'],   0, FEATURE_MAX['AR'],   0, key='AR')

    st.markdown('**Half-Time Goals**')
    c9, c10 = st.columns(2)
    HTHG = c9.number_input(FEATURE_LABELS['HTHG'], 0, FEATURE_MAX['HTHG'], 0, key='HTHG')
    HTAG = c10.number_input(FEATURE_LABELS['HTAG'], 0, FEATURE_MAX['HTAG'], 0, key='HTAG')

    predict_btn = st.button('Predict Outcome', use_container_width=True, type='primary')

with col_right:
    st.subheader('Prediction Result')

    if predict_btn:
        input_data = np.array([[HST, AST, HC, AC, HY, AY, HR, AR, HTHG, HTAG]])
        input_scaled = scaler.transform(input_data)
        proba = model.predict_proba(input_scaled)[0]
        pred_class = model.predict(input_scaled)[0]

        label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        colour_map = {0: COLOURS['A'], 1: COLOURS['D'], 2: COLOURS['H']}
        emoji_map  = {0: '🔴', 1: '🟡', 2: '🟢'}

        pred_label  = label_map[pred_class]
        pred_colour = colour_map[pred_class]
        pred_emoji  = emoji_map[pred_class]

        # Big result box
        st.markdown(
            f"<div style='background-color:{pred_colour}; padding:20px; border-radius:10px; "
            f"text-align:center;'>"
            f"<h2 style='color:white; margin:0;'>{pred_emoji} {pred_label}</h2>"
            f"<p style='color:white; margin:4px 0 0 0; font-size:14px;'>Most likely outcome</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown('')

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        labels  = ['Away Win (A)', 'Draw (D)', 'Home Win (H)']
        colours = [COLOURS['A'], COLOURS['D'], COLOURS['H']]
        bars = ax.barh(labels, [proba[0], proba[1], proba[2]],
                       color=colours, edgecolor='white', height=0.5)
        for bar, p in zip(bars, [proba[0], proba[1], proba[2]]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{p*100:.1f}%', va='center', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Probability', fontsize=10)
        ax.set_title('Outcome Probabilities', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Probability table
        prob_df = pd.DataFrame({
            'Outcome': ['Home Win (H)', 'Draw (D)', 'Away Win (A)'],
            'Probability': [f'{proba[2]*100:.1f}%', f'{proba[1]*100:.1f}%', f'{proba[0]*100:.1f}%']
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

    else:
        st.info('Fill in the match statistics on the left and click **Predict Outcome**.')

# ── Footer info ───────────────────────────────────────────────────────────────
st.divider()
with st.expander('About this model'):
    st.markdown("""
**Model:** Multinomial Logistic Regression (`sklearn`, solver=`lbfgs`, max_iter=1000)

**Training data:** 760 EPL matches from 2023/24 and 2024/25 seasons

**Features used (10):** HST, AST, HC, AC, HY, AY, HR, AR, HTHG, HTAG

**Overall accuracy:** ~62.5% (literature range for 3-class football prediction: 55–65%)

**Note on Draw predictions:** Draw F1 is ~0.22 by design. Draws are the hardest outcome
to predict in football — neither team dominates the statistics, making them statistically
ambiguous. This is a known limitation, not a model error.
    """)
