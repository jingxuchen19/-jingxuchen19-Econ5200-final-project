import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_predict

st.set_page_config(page_title="NSW Job Training: Causal Analysis", layout="wide")
st.title("Does Job Training Cause Higher Earnings?")
st.markdown("Causal analysis of the NSW program using Double Machine Learning (DML)")

# --- Load Data ---
@st.cache_data
def load_data():
    nsw = pd.read_stata("https://users.nber.org/~rdehejia/data/nsw_dw.dta")
    cps = pd.read_stata("https://users.nber.org/~rdehejia/data/cps_controls.dta")
    nsw_treated = nsw[nsw['treat'] == 1].copy()
    df = pd.concat([nsw_treated, cps], ignore_index=True)
    return df

df = load_data()

# --- Sidebar Controls ---
st.sidebar.header("Model Parameters")

model_choice = st.sidebar.selectbox(
    "Nuisance Model",
    ["Gradient Boosting", "Random Forest"]
)

n_estimators = st.sidebar.slider(
    "Number of Trees", min_value=50, max_value=300, value=100, step=50
)

st.sidebar.header("What-If Scenario")
multiplier = st.sidebar.slider(
    "Treatment Intensity Multiplier",
    min_value=0.5, max_value=3.0, value=1.0, step=0.25,
    help="What if the program effect were scaled by this factor?"
)

# --- Manual DML (no econml needed) ---
@st.cache_data
def run_dml(model_name, n_est):
    Y = df['re78'].values
    T = df['treat'].values
    covariates = ['age', 'education', 'black', 'hispanic',
                  'married', 'nodegree', 're74', 're75']
    W = df[covariates].values

    if model_name == "Gradient Boosting":
        model_y = GradientBoostingRegressor(n_estimators=n_est, random_state=42)
        model_t = GradientBoostingRegressor(n_estimators=n_est, random_state=42)
    else:
        model_y = RandomForestRegressor(n_estimators=n_est, random_state=42)
        model_t = RandomForestRegressor(n_estimators=n_est, random_state=42)

    # Step 1: Residualize Y and T
    Y_hat = cross_val_predict(model_y, W, Y, cv=5)
    T_hat = cross_val_predict(model_t, W, T, cv=5)

    Y_res = Y - Y_hat
    T_res = T - T_hat

    # Step 2: Regress Y_res on T_res (the DML estimate)
    ate = np.sum(T_res * Y_res) / np.sum(T_res * T_res)

    # Standard error
    n = len(Y)
    residuals = Y_res - ate * T_res
    denom = np.sum(T_res ** 2)
    se = np.sqrt(np.sum(residuals ** 2) / (n - 1)) / np.sqrt(denom)

    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    return ate, ci_lower, ci_upper, se

with st.spinner("Running DML estimation..."):
    ate, ci_lower, ci_upper, se = run_dml(model_choice, n_estimators)

# --- What-If Counterfactual ---
cf_ate = ate * multiplier
cf_ci_lower = ci_lower * multiplier
cf_ci_upper = ci_upper * multiplier

# --- Display Results ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("DML Estimate")
    st.metric("Average Treatment Effect", f"${ate:,.0f}")
    st.markdown(f"95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]")

with col2:
    st.subheader(f"What-If: {multiplier}x Intensity")
    st.metric("Scaled Treatment Effect", f"${cf_ate:,.0f}")
    st.markdown(f"95% CI: [${cf_ci_lower:,.0f}, ${cf_ci_upper:,.0f}]")

# --- Visualization ---
st.subheader("Estimate Comparison")

fig, ax = plt.subplots(figsize=(8, 4))

labels = ['Baseline DML', f'What-If ({multiplier}x)']
points = [ate, cf_ate]
lowers = [ci_lower, cf_ci_lower]
uppers = [ci_upper, cf_ci_upper]
errors = [[p - l for p, l in zip(points, lowers)],
          [u - p for p, u in zip(points, uppers)]]

colors = ['#1a237e', '#e65100']
for i in range(2):
    ax.errorbar(labels[i], points[i], yerr=[[errors[0][i]], [errors[1][i]]],
                fmt='o', capsize=8, markersize=10, linewidth=2, color=colors[i])

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Estimated Effect on Earnings ($)')
ax.set_title('DML Estimate vs. What-If Scenario')
plt.tight_layout()
st.pyplot(fig)

# --- Data Overview ---
st.subheader("Data Overview")
st.markdown(f"**N:** {len(df):,} observations ({df['treat'].sum():.0f} treated, {(df['treat']==0).sum():,} control)")

col3, col4 = st.columns(2)
with col3:
    st.markdown("**Balance Check (Group Means)**")
    numeric_cols = df.select_dtypes(include='number').columns
    balance = df.groupby('treat')[numeric_cols].mean().T
    balance.columns = ['Control', 'Treated']
    balance['Diff'] = balance['Treated'] - balance['Control']
    st.dataframe(balance.round(2))

with col4:
    st.markdown("**Earnings Distribution by Group**")
    fig2, ax2 = plt.subplots()
    df[df['treat']==0]['re78'].hist(bins=50, alpha=0.5, label='Control', ax=ax2, density=True)
    df[df['treat']==1]['re78'].hist(bins=50, alpha=0.5, label='Treated', ax=ax2, density=True)
    ax2.set_xlabel('Earnings in 1978 ($)')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
