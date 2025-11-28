import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global Constants (Must match streamlit_app.py) ---
MODEL_PATH = "xgboost_model.json" 
FINAL_FEATURES = [ 
    # NOTE: PASTE THE SAME EXACT LIST OF FEATURES HERE
    'annual_income', 
    'loan_amount', 
    'cibil_score', 
    'income_to_loan_ratio', 
    'residue_time_to_loan_end',
    'is_short_term_loan',
    'TE_loan_purpose', 
    'TE_education', 
    # ... all 19 features ...
]

# --- 1. CACHED RESOURCES ---

@st.cache_resource
def load_model(path):
    """Loads the trained XGBoost model once."""
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model

@st.cache_data
def generate_synthetic_data(num_samples=1000):
    """
    Creates a small, synthetic dataset based on the feature list 
    to be used for global SHAP analysis and plotting.
    (Since the real training data is not in the app)
    """
    data = {}
    for feature in FINAL_FEATURES:
        if feature.startswith('TE_'):
            # For Target-Encoded features, generate values near the global mean (e.g., 0.8)
            data[feature] = np.random.normal(loc=0.84, scale=0.05, size=num_samples)
        else:
            # For numerical features, generate typical ranges
            if 'income' in feature:
                data[feature] = np.random.normal(loc=70000, scale=30000, size=num_samples)
            elif 'loan_amount' in feature:
                data[feature] = np.random.uniform(5000, 40000, size=num_samples)
            elif 'cibil_score' in feature:
                data[feature] = np.random.randint(600, 850, size=num_samples)
            elif 'is_short_term_loan' in feature:
                data[feature] = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
            else:
                data[feature] = np.random.rand(num_samples) # Fallback random data

    return pd.DataFrame(data)


# --- 2. MAIN PAGE FUNCTION ---

def app():
    st.title("📊 Global Model Insights & Feature Analysis")
    st.markdown("---")

    model = load_model(MODEL_PATH)
    data_sample = generate_synthetic_data()

    # --- Section 1: Global Feature Importance (SHAP) ---
    st.header("1. Global Model Structure (SHAP Summary)")
    st.info("The SHAP summary shows the feature importance across the entire dataset. Features higher up are more important, and colors indicate their impact.")

    # Calculate SHAP values for the synthetic dataset
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_sample)

    # Plot the SHAP Summary Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values[1], data_sample, show=False, max_display=15) # [1] for the positive class
    st.pyplot(fig)

    # --- Section 2: Feature Relationship Plots (New Charts) ---
    st.header("2. Key Feature Relationships")
    
    # We use the model to predict on the synthetic data to get a target probability
    data_sample['Predicted_Risk'] = model.predict_proba(data_sample)[:, 0] # 0 = default risk
    
    col1, col2 = st.columns(2)
    
    # Chart A: Risk vs. Income/Score (Scatter Plot)
    with col1:
        st.subheader("Predicted Risk by Income vs. Loan Amount")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x='annual_income', 
            y='loan_amount', 
            hue='Predicted_Risk', 
            palette="viridis", 
            data=data_sample, 
            ax=ax
        )
        ax.set_title("Predicted Risk Distribution (Synthetic Data)")
        st.pyplot(fig)
        

    # Chart B: Feature Distribution (Histogram)
    with col2:
        st.subheader("Distribution of CIBIL Score")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data_sample['cibil_score'], kde=True, bins=30, color='skyblue', ax=ax)
        ax.axvline(data_sample['cibil_score'].mean(), color='red', linestyle='--', label='Mean Score')
        ax.set_title("Distribution of CIBIL Scores")
        st.pyplot(fig)
        


if __name__ == "__main__":
    app()
