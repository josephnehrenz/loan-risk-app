import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global Constants ---
MODEL_PATH = "xgboost_model.json"
FINAL_FEATURES = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate',
    'age',
    'monthly_income',
    'loan_term',
    'installment',
    'num_of_open_accounts',
    'total_credit_limit',
    'current_balance',
    'delinquency_history',
    'public_records',
    'num_of_delinquencies',
    'income_loan_ratio',
    'loan_to_income',
    'total_debt',
    'available_income',
    'monthly_payment_approx',
    'payment_to_income',
    'default_risk_score',
    'grade_number',
    'TE_gender',
    'TE_marital_status',
    'TE_education_level',
    'TE_employment_status',
    'TE_loan_purpose',
    'TE_grade_subgrade',
    'TE_grade_letter'
]
GLOBAL_MEAN_TARGET = 0.798820
NUMERICAL_STATS_MAPPING = { 
    'annual_income': {'mean': -0.00, 'std': 1.00, 'min': -1.58, 'max': 12.92},
    'debt_to_income_ratio': {'mean': -0.00, 'std': 1.00, 'min': -1.60, 'max': 7.38},
    'credit_score': {'mean': 0.00, 'std': 1.00, 'min': -5.16, 'max': 3.03},
    'loan_amount': {'mean': 0.00, 'std': 1.00, 'min': -2.10, 'max': 4.90},
    'interest_rate': {'mean': -0.00, 'std': 1.00, 'min': -4.56, 'max': 4.30},
    'income_loan_ratio': {'mean': -0.00, 'std': 1.00, 'min': -0.56, 'max': 44.59},
    'loan_to_income': {'mean': 0.00, 'std': 1.00, 'min': -1.16, 'max': 13.75},
    'total_debt': {'mean': -0.00, 'std': 1.00, 'min': -1.15, 'max': 25.68},
    'available_income': {'mean': 0.00, 'std': 1.00, 'min': -1.65, 'max': 14.12},
    'monthly_payment_approx': {'mean': 0.00, 'std': 1.00, 'min': -1.99, 'max': 7.21},
    'payment_to_income': {'mean': -0.00, 'std': 1.00, 'min': -1.12, 'max': 14.06},
    'default_risk_score': {'mean': -0.00, 'std': 1.00, 'min': -3.07, 'max': 6.90},
    'grade_number': {'mean': 0.00, 'std': 1.00, 'min': -1.42, 'max': 1.43}
}


# --- 1. CACHED RESOURCES ---

@st.cache_resource
def load_model(path):
    """Loads the trained XGBoost model once."""
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model

@st.cache_data
def generate_synthetic_data(num_samples=1000):
    data = {}
    for feature in FINAL_FEATURES:
        if feature.startswith('TE_'):
            # Use the global mean for target-encoded features
            data[feature] = np.random.normal(loc=GLOBAL_MEAN_TARGET, scale=0.05, size=num_samples)
            
        elif feature in NUMERICAL_STATS_MAPPING:
            # Use the actual mean and std from your scaled data
            stats = NUMERICAL_STATS_MAPPING[feature]
            data[feature] = np.random.normal(loc=stats['mean'], scale=stats['std'], size=num_samples)
            
            # Optional: Clip values to stay within the min/max range for realism
            data[feature] = np.clip(data[feature], stats['min'], stats['max'])

        else:
            data[feature] = np.random.normal(loc=0.0, scale=1.0, size=num_samples)

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

    # Check if SHAP returned a list (standard for binary classification)
    if isinstance(shap_values, list):
        # We target the positive class (index 1) which is the full matrix (N_samples, N_features)
        shap_matrix = shap_values[1]
    else:
        # If it's not a list, it must be the single matrix (N_samples, N_features)
        shap_matrix = shap_values
        
    # Plot the SHAP Summary Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # Pass the confirmed 2D matrix to the plot function
    shap.summary_plot(shap_matrix, data_sample, show=False, max_display=15) 
    st.pyplot(fig) 

    # --- Section 2: Feature Relationship Plots (New Charts) ---
    st.header("2. Key Feature Relationships")
    
    # We use the model to predict on the synthetic data to get a target probability
    # 0 = default risk (Predicted_Risk is the probability of default)
    data_sample['Predicted_Risk'] = model.predict_proba(data_sample)[:, 0] 
    
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
        # Ensure 'credit_score' is used as 'cibil_score' wasn't in the final features list.
        st.subheader("Distribution of Credit Score")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data_sample['credit_score'], kde=True, bins=30, color='skyblue', ax=ax)
        ax.axvline(data_sample['credit_score'].mean(), color='red', linestyle='--', label='Mean Score')
        ax.set_title("Distribution of Credit Scores (Scaled)")
        st.pyplot(fig) 

[Image of a histogram plot showing the frequency distribution of Credit scores]



if __name__ == "__main__":
    app()
