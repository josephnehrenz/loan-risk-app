import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION AND CONSTANTS (MOCK DATA FOR ENGINEERING) ---
MODEL_PATH = "xgboost_model.json"

# Mock constants derived from the training set statistics (required for feature engineering and scaling)
NUMERICAL_STATS = {
    'annual_income': {'mean': 80000, 'std': 50000},
    'debt_to_income_ratio': {'mean': 15, 'std': 10},
    'credit_score': {'mean': 700, 'std': 50},
    'loan_amount': {'mean': 15000, 'std': 10000},
    'interest_rate': {'mean': 12, 'std': 5}
}

# Mock Target Encoding (TE) values derived from the training set
TE_MAPPINGS = {
    'gender': {'Female': 0.7, 'Male': 0.5, 'Other': 0.6},
    'marital_status': {'Single': 0.55, 'Married': 0.65, 'Divorced': 0.45},
    'education_level': {'High School': 0.5, 'Bachelor': 0.6, 'Master': 0.7, 'PhD': 0.75},
    'employment_status': {'Employed': 0.7, 'Self-Employed': 0.6, 'Retired': 0.8, 'Unemployed': 0.4},
    'loan_purpose': {'Debt Consolidation': 0.55, 'Home Improvement': 0.65, 'Other': 0.5, 'Medical': 0.45},
    'grade_subgrade': {'A1': 0.85, 'B3': 0.75, 'C5': 0.65, 'D1': 0.55, 'F3': 0.45},
    'grade_letter': {'A': 0.8, 'B': 0.7, 'C': 0.6, 'D': 0.5, 'E': 0.4}
}

# Default Input Values (Used on initial page load - Request 2)
DEFAULT_INPUTS = {
    'annual_income': 55000.0,
    'debt_to_income_ratio': 15.0,
    'credit_score': 720,
    'loan_amount': 12000.0,
    'interest_rate': 8.0,
    'gender': 'Male',
    'marital_status': 'Married',
    'education_level': 'Bachelor',
    'employment_status': 'Employed',
    'loan_purpose': 'Debt Consolidation',
    'grade_letter': 'B',
    'grade_subgrade': 'B3'
}

# DEFINITIVE LIST OF ALL 20 FEATURES (This is the order required by the model)
FINAL_FEATURES = [
    'annual_income', 
    'debt_to_income_ratio', 
    'credit_score', 
    'loan_amount', 
    'interest_rate', 
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

# --- 2. CACHED MODEL LOADING ---
@st.cache_resource
def load_model(path):
    """Loads the trained XGBoost model once."""
    try:
        model = xgb.XGBClassifier()
        # Your model will load here once xgboost_model.json is present in the environment
        model.load_model(path) 
        return model
    except Exception:
        # FALLBACK: If the real model file is missing, we use a mock object.
        st.error("Model file (xgboost_model.json) not found. Using a mock predictor.")
        class MockModel:
            def predict_proba(self, X):
                # Returns a repayment probability of 65% for any input
                return np.array([[0.35, 0.65]])
        return MockModel()

# --- 3. FEATURE ENGINEERING AND PREDICTION ---
def get_prediction_and_explanation(model, raw_inputs):
    # 1. Standard Feature Engineering (as done in the original notebook)
    data = {}
    
    # Simple Calculated Features
    data['income_loan_ratio'] = raw_inputs['annual_income'] / (raw_inputs['loan_amount'] + 1e-6)
    data['loan_to_income'] = raw_inputs['loan_amount'] / (raw_inputs['annual_income'] + 1e-6)
    data['total_debt'] = raw_inputs['annual_income'] * (raw_inputs['debt_to_income_ratio'] / 100)
    data['available_income'] = raw_inputs['annual_income'] - data['total_debt']
    data['monthly_payment_approx'] = (raw_inputs['loan_amount'] * (raw_inputs['interest_rate'] / 1200)) / (1 - (1 + raw_inputs['interest_rate'] / 1200)**(-60)) if raw_inputs['interest_rate'] > 0 else raw_inputs['loan_amount'] / 60
    data['payment_to_income'] = data['monthly_payment_approx'] / (raw_inputs['annual_income'] / 12 + 1e-6)
    data['default_risk_score'] = 1000 - raw_inputs['credit_score'] # Inverse credit score
    
    # 2. Target Encoding Features
    for key, mapping in TE_MAPPINGS.items():
        te_key = f'TE_{key}'
        data[te_key] = mapping.get(raw_inputs[key], 0.5) # Default to 0.5 if value not found

    # 3. Grade Number (A=1, B=2, ...)
    data['grade_number'] = ord(raw_inputs['grade_letter']) - ord('A') + 1

    # Combine with original numerical inputs
    final_input_data = {
        'annual_income': raw_inputs['annual_income'],
        'debt_to_income_ratio': raw_inputs['debt_to_income_ratio'],
        'credit_score': raw_inputs['credit_score'],
        'loan_amount': raw_inputs['loan_amount'],
        'interest_rate': raw_inputs['interest_rate'],
        **data
    }
    
    # Create DataFrame for model input, ensuring correct order
    X_single = pd.DataFrame([final_input_data])[FINAL_FEATURES]
    
    # 4. Prediction
    prediction_proba = model.predict_proba(X_single)[0, 1] # Probability of repayment (class 1)
    
    # 5. SHAP Explanation
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_single)
        # We focus on SHAP values for class 1 (repayment)
        shap_values_class1 = shap_values[1] 
    except Exception as e:
        st.warning(f"Using Mock SHAP Explanation due to Explainer Error: {e}")
        # Fallback for mock model or if explainer fails
        shap_values_class1 = np.random.randn(1, len(FINAL_FEATURES)) * 0.1
        
    return prediction_proba, X_single, shap_values_class1

# --- 4. VISUALIZATION HELPERS ---
def plot_shap_waterfall(X_single, shap_values_class1, expected_value=0.6):
    
    # Create a dummy explainer structure for the waterfall plot
    class DummyExplainer:
        def __init__(self, expected_value):
            self.expected_value = expected_value

    fig, ax = plt.subplots(figsize=(8, 6)) # Adjusted height for better balance
    
    # We need to map the feature values from the single observation X_single 
    # back to the SHAP values array.
    shap_object = shap.Explanation(
        values=shap_values_class1[0],
        base_values=expected_value,
        data=X_single.iloc[0].values,
        feature_names=FINAL_FEATURES
    )
    
    shap.waterfall_plot(shap_object, max_display=10, show=False)
    ax.set_title("Feature Contribution to Repayment Score")
    ax.set_xlabel("Log-Odds of Repayment") # SHAP default x-label
    
    return fig

def display_gauge(proba):
    # Determine color and icon based on probability
    if proba >= 0.75:
        color = "green"
        icon = "✅"
        risk_level = "Low Risk"
    elif proba >= 0.60:
        color = "orange"
        icon = "⚠️"
        risk_level = "Moderate Risk"
    else:
        color = "red"
        icon = "🛑"
        risk_level = "High Risk"

    score_percent = int(proba * 100)
    
    st.markdown(f"""
    <div style="text-align: center; border: 2px solid {color}; padding: 10px; border-radius: 10px;">
        <h3>{icon} Predicted Repayment Score</h3>
        <h1 style='color: {color}; font-size: 60px;'>{score_percent}%</h1>
        <p style='font-weight: bold; font-size: 18px;'>Risk Level: {risk_level}</p>
    </div>
    """, unsafe_allow_html=True)
    
    return risk_level

# --- 5. MAIN APPLICATION LAYOUT ---
def main():
    st.set_page_config(layout="wide")

    st.title("💰 Loan Risk Advisor: Applicant Prediction")
    st.markdown("Use the controls on the left to simulate a loan applicant and see the real-time prediction and explanation.")

    # --- INPUT SIDEBAR (REQUEST 1: TWO COLUMNS) ---
    with st.sidebar:
        st.header("Applicant Profile & Loan Terms")
        st.markdown("---")
        
        # Two columns for financial inputs (Request 1)
        st.subheader("Financial Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=DEFAULT_INPUTS['annual_income'], step=5000.0)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, value=DEFAULT_INPUTS['loan_amount'], step=1000.0)
        
        with col2:
            credit_score = st.number_input("Credit Score (300-850)", min_value=300, max_value=850, value=DEFAULT_INPUTS['credit_score'], step=1)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=DEFAULT_INPUTS['interest_rate'], step=0.1)

        debt_to_income_ratio = st.slider("Debt-to-Income Ratio (%)", min_value=0.0, max_value=50.0, value=DEFAULT_INPUTS['debt_to_income_ratio'], step=0.1)
        st.markdown("---")

        # Two columns for categorical inputs (Request 1)
        st.subheader("Profile & Loan Details")
        col3, col4 = st.columns(2)

        with col3:
            gender = st.selectbox("Gender", options=list(TE_MAPPINGS['gender'].keys()), index=list(TE_MAPPINGS['gender'].keys()).index(DEFAULT_INPUTS['gender']))
            marital_status = st.selectbox("Marital Status", options=list(TE_MAPPINGS['marital_status'].keys()), index=list(TE_MAPPINGS['marital_status'].keys()).index(DEFAULT_INPUTS['marital_status']))
            grade_letter = st.selectbox("Loan Grade (A-F)", options=list(TE_MAPPINGS['grade_letter'].keys()), index=list(TE_MAPPINGS['grade_letter'].keys()).index(DEFAULT_INPUTS['grade_letter']))

        with col4:
            education_level = st.selectbox("Education Level", options=list(TE_MAPPINGS['education_level'].keys()), index=list(TE_MAPPINGS['education_level'].keys()).index(DEFAULT_INPUTS['education_level']))
            employment_status = st.selectbox("Employment Status", options=list(TE_MAPPINGS['employment_status'].keys()), index=list(TE_MAPPINGS['employment_status'].keys()).index(DEFAULT_INPUTS['employment_status']))
            loan_purpose = st.selectbox("Loan Purpose", options=list(TE_MAPPINGS['loan_purpose'].keys()), index=list(TE_MAPPINGS['loan_purpose'].keys()).index(DEFAULT_INPUTS['loan_purpose']))
            # Subgrade input is often optional/derived, use a default selection
            grade_subgrade = st.selectbox("Loan Subgrade (A1, B3, etc.)", options=list(TE_MAPPINGS['grade_subgrade'].keys()), index=list(TE_MAPPINGS['grade_subgrade'].keys()).index(DEFAULT_INPUTS['grade_subgrade']))


    # --- MAIN CONTENT LOGIC ---
    
    # Load Model (or Mock Model)
    model = load_model(MODEL_PATH)

    # 1. Gather all inputs
    raw_inputs = {
        'annual_income': annual_income,
        'debt_to_income_ratio': debt_to_income_ratio,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'gender': gender,
        'marital_status': marital_status,
        'education_level': education_level,
        'employment_status': employment_status,
        'loan_purpose': loan_purpose,
        'grade_letter': grade_letter,
        'grade_subgrade': grade_subgrade
    }

    # 2. Get Prediction and SHAP Explanation
    prediction_proba, X_single, shap_values_class1 = get_prediction_and_explanation(model, raw_inputs)
    
    # --- MAIN CONTENT LAYOUT ---
    
    col_score, col_metrics = st.columns([1, 2])
    
    # COLUMN 1: PREDICTION GAUGE AND SUMMARY
    with col_score:
        display_gauge(prediction_proba)
        
        risk_level = display_gauge(prediction_proba)

        st.markdown(f"### Interpretation")
        st.markdown(f"""
        Based on the applicant profile, the model predicts a **{int(prediction_proba * 100)}%** chance of successful loan repayment.
        
        This places the application in the **{risk_level}** category. The detailed breakdown on the right shows which specific input factors pushed the score higher or lower.
        """)
    
    # COLUMN 2: APPLICANT METRICS (REQUEST 3: TWO COLUMNS)
    with col_metrics:
        st.markdown("### Applicant & Loan Metrics Summary")
        
        # Two columns for the summary display (Request 3)
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.metric("Annual Income", f"${annual_income:,.0f}")
            st.metric("Loan Amount", f"${loan_amount:,.0f}")
            st.metric("DTI Ratio", f"{debt_to_income_ratio:.1f}%")
            st.metric("Interest Rate", f"{interest_rate:.1f}%")

        with col_m2:
            st.metric("Credit Score", f"{credit_score}")
            st.metric("Employment Status", employment_status)
            st.metric("Education Level", education_level)
            st.metric("Loan Grade", grade_letter)
            
        st.markdown("---")


    st.markdown("## Individual Prediction Explanation (SHAP Waterfall Plot)")
    st.info("The Waterfall plot shows how each feature (in the input sidebar) contributes to the final prediction score, relative to the model's average expected value.")
    
    # SHAP PLOT (Adjusted to be visually equal to the combined height of the sections above)
    # The sections above are the Score (approx height 6) + Metrics (approx height 4) -> Total height ~10
    # Waterfall plot is large, so 7 should balance well with the vertical space added by titles/borders.
    try:
        fig_waterfall = plot_shap_waterfall(X_single, shap_values_class1, expected_value=0.6)
        st.pyplot(fig_waterfall, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate SHAP Waterfall Plot: {e}")
        
# Run the main function
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Simple error handling for non-streamlit environment
        pass
