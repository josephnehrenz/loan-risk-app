import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json # Used for error handling model loading

# --- 1. CONFIGURATION AND CONSTANTS ---

st.set_page_config(
    page_title="Loan Risk Advisor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# A. GLOBAL MEAN TARGET
GLOBAL_MEAN_TARGET = 0.798820 

# B. FINAL FEATURES LIST (20 features that the model was trained on)
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

# C. FEATURE CATEGORY LISTS
NUMERICAL_FEATURES = [
    'annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate', 
    'income_loan_ratio', 'loan_to_income', 'total_debt', 'available_income', 
    'monthly_payment_approx', 'payment_to_income', 'default_risk_score', 'grade_number'
]

# D. TARGET ENCODING MAPPINGS (Added 'grade_letter' mapping since the TE_grade_letter feature is required)
TE_MAPPINGS = {
    'gender': {'Female': 0.8017, 'Male': 0.7958, 'Other': 0.7953},
    'marital_status': {'Divorced': 0.7966, 'Married': 0.7991, 'Single': 0.7989, 'Widowed': 0.7898},
    'education_level': {"Bachelor's": 0.7889, 'High School': 0.8097, "Master's": 0.8023, 'Other': 0.8028, 'PhD': 0.8301},
    'employment_status': {'Employed': 0.8941, 'Retired': 0.9972, 'Self-employed': 0.8985, 'Student': 0.2635, 'Unemployed': 0.0776},
    'loan_purpose': {'Business': 0.8131, 'Car': 0.8006, 'Debt consolidation': 0.7969, 'Education': 0.7771, 'Home': 0.8232, 'Medical': 0.7781, 'Other': 0.8024, 'Vacation': 0.7961},
    'grade_subgrade': {'A1': 0.9525, 'A2': 0.9529, 'A3': 0.9555, 'A4': 0.9571, 'A5': 0.945, 'B1': 0.9163, 'B2': 0.9374, 'B3': 0.94, 'B4': 0.9318, 'B5': 0.9342, 'C1': 0.8601, 'C2': 0.8512, 'C3': 0.836, 'C4': 0.844, 'C5': 0.8463, 'D1': 0.7319, 'D2': 0.721, 'D3': 0.696, 'D4': 0.7147, 'D5': 0.713, 'E1': 0.652, 'E2': 0.6627, 'E3': 0.6418, 'E4': 0.6496, 'E5': 0.6695, 'F1': 0.6245, 'F2': 0.6177, 'F3': 0.6041, 'F4': 0.637, 'F5': 0.6393},
    'grade_letter': {'A': 0.954, 'B': 0.935, 'C': 0.845, 'D': 0.718, 'E': 0.655, 'F': 0.625, 'G': 0.580} # Estimated values
}

# The list of non-TE features that need input sliders (Must match the 13 numerical features)
ALL_SLIDER_FEATURES = NUMERICAL_FEATURES 

MODEL_PATH = "xgboost_model.json"


# --- 2. CACHED MODEL LOADING & PREDICTION ---

@st.cache_resource
def load_model(path):
    """Loads the trained XGBoost model once."""
    try:
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        st.stop()

def predict_and_explain(model, input_data):
    """Makes a prediction and calculates SHAP values. (FIXED: Robust handling of single-sample SHAP output)"""
    
    # Reorder columns to match the trained model's feature order
    input_data = input_data[FINAL_FEATURES]
    
    # 1. Prediction (returns probability of TARGET=1, i.e., loan paid back)
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    
    # 2. SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(input_data)
    
    # 3. Final Return Values (ROBUST INDEXING ADDED HERE)
    
    if isinstance(shap_values_raw, list):
        # Standard binary classification output: [class_0 values, class_1 values]
        # We need the values for class 1 (repayment)
        shap_values_matrix = shap_values_raw[1]
        expected_value = explainer.expected_value[1]
    else:
        # Simplified output (single array, assumed to be the positive class)
        shap_values_matrix = shap_values_raw
        expected_value = explainer.expected_value 

    # Ensure the SHAP values are a 1D vector (N_features,) for plotting
    # This correctly extracts the single row of values whether it's (1, 20) or (20,)
    shap_values = np.array(shap_values_matrix).flatten()
    
    return prediction_proba, shap_values, expected_value


# --- 3. APP LAYOUT AND LOGIC (The Home Page) ---

def main():
    model = load_model(MODEL_PATH)
    
    st.title("💰 Loan Risk Advisor: Prediction Tool")
    st.markdown("---")

    st.sidebar.header("Input Loan Application Details")
    user_input = {}
    
    # --- A. Input Sliders (Numerical Features) ---
    st.sidebar.subheader("1. Financial and Profile Metrics (Scaled Data)")
    st.sidebar.caption("Input data is scaled. 0.0 is the mean value.")
    
    # We loop through all non-TE features (now 13 features)
    for feature in ALL_SLIDER_FEATURES:
        max_val = 5.0 # Use a safe default since the true max/min are unknown in this file
        
        user_input[feature] = st.sidebar.slider(
            f"{feature.replace('_', ' ').title()}", 
            min_value=-5.0, 
            max_value=max_val,
            value=0.0, # Default to the mean (0.0 after scaling)
            step=0.01
        )

    # --- B. Select Boxes (Target-Encoded Features) ---
    st.sidebar.subheader("2. Categorical Features")
    
    # The TE_MAPPINGS dictionary now includes 'grade_letter', which is CRITICAL
    for original_col, mapping in TE_MAPPINGS.items():
        te_feature_name = f'TE_{original_col}'
        
        selected_option = st.sidebar.selectbox(
            f"Select {original_col.replace('_', ' ').title()}", 
            list(mapping.keys())
        )
        # Map the selected option back to the numerical TE value
        user_input[te_feature_name] = mapping[selected_option]

    # --- C. Prediction Button and Results ---
    
    if st.sidebar.button("Analyze Loan Risk", type="primary"):
        st.subheader("Analysis Results")
        
        # 1. Convert user inputs into the DataFrame 
        input_df = pd.DataFrame([user_input])
        
        # 2. Reorder columns to match the trained model's feature order
        try:
            # This line should now succeed, as input_df and FINAL_FEATURES match the model
            input_df = input_df[FINAL_FEATURES]
        except KeyError as e:
            st.error(f"Input mismatch! Check your FINAL_FEATURES list: Missing feature {e}. (Did you forget to add the 'grade_letter' selectbox and mapping?)")
            return

        # Get prediction and SHAP values
        probability, shap_values, expected_value = predict_and_explain(model, input_df)
        risk_percentage = probability * 100
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                label="Probability of Repayment (Low Risk)", 
                value=f"{risk_percentage:.1f}%", 
                delta=f"{(probability - GLOBAL_MEAN_TARGET) * 100:.1f} pts vs Global Average"
            )
            
            # Simple assessment text
            if probability > 0.85:
                st.success("High Likelihood of Repayment. 👍")
            elif probability < 0.70:
                st.warning("Higher Risk of Default. 🛑")
            else:
                st.info("Moderate Risk Profile. ⚠️")

        # Display the SHAP Explanation (Local Interpretability)
        with col2:
            st.write("#### Feature Contribution for This Loan Applicant")
            st.info("The chart shows which features pushed the prediction toward repayment (Blue) or default (Red).")
            
            # Matplotlib/SHAP rendering
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values,
                    base_values=expected_value,
                    data=input_df.iloc[0],
                    feature_names=FINAL_FEATURES
                ),
                max_display=10,
                show=False
            )
            st.pyplot(fig) 

# Run the main function
if __name__ == "__main__":
    main()
