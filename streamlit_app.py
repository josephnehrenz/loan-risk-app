import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json # Used for error handling model loading

# --- 1. CONFIGURATION AND CONSTANTS ---

# Streamlit page configuration is handled in the main function for better control
# over the application lifecycle and layout
# st.set_page_config is called inside the `main` function below.

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

# D. TARGET ENCODING MAPPINGS
TE_MAPPINGS = {
    'gender': {'Female': 0.8017, 'Male': 0.7958, 'Other': 0.7953},
    'marital_status': {'Divorced': 0.7966, 'Married': 0.7991, 'Single': 0.7989, 'Widowed': 0.7898},
    'education_level': {"Bachelor's": 0.7889, 'High School': 0.8097, "Master's": 0.8023, 'Other': 0.8028, 'PhD': 0.8301},
    'employment_status': {'Employed': 0.8941, 'Retired': 0.9972, 'Self-employed': 0.8985, 'Student': 0.2635, 'Unemployed': 0.0776},
    'loan_purpose': {'Business': 0.8131, 'Car': 0.8006, 'Debt consolidation': 0.7969, 'Education': 0.7771, 'Home': 0.8232, 'Medical': 0.7781, 'Other': 0.8024, 'Vacation': 0.7961},
    'grade_subgrade': {'A1': 0.9525, 'A2': 0.9529, 'A3': 0.9555, 'A4': 0.9571, 'A5': 0.945, 'B1': 0.9163, 'B2': 0.9374, 'B3': 0.94, 'B4': 0.9318, 'B5': 0.9342, 'C1': 0.8601, 'C2': 0.8512, 'C3': 0.836, 'C4': 0.844, 'C5': 0.8463, 'D1': 0.7319, 'D2': 0.721, 'D3': 0.696, 'D4': 0.7147, 'D5': 0.713, 'E1': 0.652, 'E2': 0.6627, 'E3': 0.6418, 'E4': 0.6496, 'E5': 0.6695, 'F1': 0.6245, 'F2': 0.6177, 'F3': 0.6041, 'F4': 0.637, 'F5': 0.6393},
    'grade_letter': {'A': 0.954, 'B': 0.935, 'C': 0.845, 'D': 0.718, 'E': 0.655, 'F': 0.625, 'G': 0.580}
}

ALL_SLIDER_FEATURES = NUMERICAL_FEATURES 
MODEL_PATH = "xgboost_model.json"


# --- 2. CACHED MODEL LOADING & PREDICTION ---

@st.cache_resource
def load_model(path):
    """Loads the trained XGBoost model once."""
    try:
        model = xgb.XGBClassifier()
        # Mock load for environments without the file, ensuring the app runs
        try:
            model.load_model(path)
        except xgb.core.XGBoostError:
            st.warning("Mocking model: Real XGBoost model file not found. Predictions will be constant.")
            class MockModel:
                def predict_proba(self, X):
                    # Returns a constant repayment probability of 80% for mock
                    return np.array([[0.20, 0.80]])
                @property
                def expected_value(self):
                    return [0.0, 0.79] # Mock expected values for SHAP
            return MockModel()
        return model
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        st.stop()

def predict_and_explain(model, input_data):
    """Makes a prediction and calculates SHAP values."""
    
    # Reorder columns to match the trained model's feature order
    input_data = input_data[FINAL_FEATURES]
    
    # 1. Prediction (returns probability of TARGET=1, i.e., loan paid back)
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    
    # 2. SHAP Explanation
    try:
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(input_data)
        
        if isinstance(shap_values_raw, list):
            # Standard binary classification output: [class_0 values, class_1 values]
            shap_values_matrix = shap_values_raw[1]
            expected_value = explainer.expected_value[1]
        else:
            # Simplified output
            shap_values_matrix = shap_values_raw
            expected_value = explainer.expected_value 

        shap_values = np.array(shap_values_matrix).flatten()
        
    except Exception:
        # Fallback if explainer fails (e.g., mock model in use)
        shap_values = np.zeros(len(FINAL_FEATURES))
        expected_value = 0.79 # Fallback value
        if hasattr(model, 'expected_value'):
             expected_value = model.expected_value[1] if isinstance(model.expected_value, list) else model.expected_value
        
    return prediction_proba, shap_values, expected_value


# --- 3. PAGE RENDERING FUNCTIONS ---

def render_prediction_page(model):
    """Renders the Applicant Prediction page content."""
    
    st.sidebar.header("Input Loan Application Details")
    user_input = {}
    
    # --- Custom CSS for Blue Slider (Kept) ---
    st.sidebar.markdown("""
        <style>
        /* Target the slider track fill */
        .stSlider > div > div > div:nth-child(2) {
            background-color: #1E90FF !important; /* Blue color */
        }
        /* Target the slider handle */
        .stSlider > div > div > div:nth-child(2) > div:nth-child(1) {
            background-color: #1E90FF !important; /* Blue color for the fill */
        }
        </style>
    """, unsafe_allow_html=True)
    
    # --- A. Input Sliders (Numerical Features) - Two Column Layout (Kept) ---
    st.sidebar.subheader("1. Financial and Profile Metrics (Scaled Data)")
    st.sidebar.caption("Input data is scaled. 0.0 is the mean value.")
    
    col_num1, col_num2 = st.sidebar.columns(2)
    
    # We loop through all non-TE features (13 features)
    for i, feature in enumerate(ALL_SLIDER_FEATURES):
        max_val = 5.0 
        
        # Split features across the two columns (7 in col1, 6 in col2)
        target_column = col_num1 if i < 7 else col_num2 
        
        user_input[feature] = target_column.slider(
            f"{feature.replace('_', ' ').title()}", 
            min_value=-5.0, 
            max_value=max_val,
            value=0.0, # Default to the mean (0.0 after scaling)
            step=0.01
        )
    
    # --- B. Select Boxes (Target-Encoded Features) - Single Column Layout (Reverted) ---
    st.sidebar.subheader("2. Categorical Features")
    
    # Using st.sidebar directly for single column
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
        
        # 2. Validation and Prediction
        try:
            # Ensure the input DF has the correct feature order for the model
            input_df = input_df[FINAL_FEATURES]
        except KeyError as e:
            st.error(f"Input mismatch! Missing feature {e}.")
            return

        # Get prediction and SHAP values
        probability, shap_values, expected_value = predict_and_explain(model, input_df)
        risk_percentage = probability * 100
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Larger, blue probability display
            st.markdown(
                f"<h2 style='color: #1E90FF; font-weight: bold;'>Repayment Score:</h2>"
                f"<h1 style='color: #1E90FF; font-weight: bolder;'>{risk_percentage:.1f}%</h1>", 
                unsafe_allow_html=True
            )
            
            # Metric for delta (smaller)
            st.metric(
                label="vs Global Average", 
                value="", 
                delta=f"{(probability - GLOBAL_MEAN_TARGET) * 100:.1f} pts"
            )
            
            # Simple assessment text
            if probability > 0.85:
                st.success("High Likelihood of Repayment. 👍")
            elif probability < 0.70:
                st.warning("Higher Risk of Default. 🛑")
            else:
                st.info("Moderate Risk Profile. ⚠️")

            st.markdown("---")
            st.write("#### Key Input Summary")
            
            # Summarize inputs to balance the column
            st.markdown("##### Financial Metrics (Scaled)")
            st.markdown(f"**Annual Income:** `{user_input['annual_income']:.2f}`")
            st.markdown(f"**Credit Score:** `{user_input['credit_score']:.2f}`")
            st.markdown(f"**Loan Amount:** `{user_input['loan_amount']:.2f}`")
            st.markdown(f"**Interest Rate:** `{user_input['interest_rate']:.2f}`")
            
            st.markdown("---")
            st.markdown("##### Profile Metrics (Selected)")
            
            # Categorical Inputs summary
            for original_col in ['gender', 'employment_status', 'loan_purpose', 'grade_letter']:
                te_value = user_input[f'TE_{original_col}']
                # Find the key (label) corresponding to the TE value
                selected_label = next(
                    (key for key, value in TE_MAPPINGS[original_col].items() if value == te_value),
                    "N/A"
                )
                st.markdown(f"**{original_col.replace('_', ' ').title()}:** `{selected_label}`")

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
            
# --- Mock Pages for Navigation ---
def render_insights_page():
    st.header("📈 Model Insights")
    st.markdown("This page would display global feature importance, model bias metrics, and historical performance metrics over time (e.g., AUC, F1-Score).")
    st.info("To implement this, you would need to calculate global SHAP values or use permutation importance on a large dataset and visualize the results here.")

def render_about_page():
    st.header("ℹ️ About the Loan Risk Advisor")
    st.markdown("""
    This application serves as an interactive demonstration of a Machine Learning interpretability platform for loan risk assessment.
    
    - **Model:** The core is an XGBoost Classifier trained to predict the probability of successful loan repayment (Target=1).
    - **Features:** Inputs are scaled (numerical features centered at 0.0) or Target Encoded (categorical features replaced by their historical repayment rate).
    - **Explanation:** The **SHAP Waterfall Plot** provides a local explanation, showing how each specific input parameter moves the prediction score away from the global average.
    """)

# --- 4. TOP-LEVEL APP ROUTER ---
def main():
    st.set_page_config(
        page_title="Loan Risk Advisor", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )

    # --- TOP RIBBON NAVIGATION (Implemented using st.radio) ---
    PAGES = ["Applicant Prediction", "Insights", "About"]
    
    # Custom CSS to make the radio buttons look like a clean top navigation bar
    st.markdown("""
        <style>
        /* Make Radio Buttons Horizontal */
        div.row-widget.stRadio > div {
            flex-direction: row;
            justify-content: start; /* Align to the left */
            gap: 20px;
        }
        /* Style the labels to look like links/tabs */
        div.stRadio > label {
            padding: 5px 15px;
            border-bottom: 2px solid transparent;
            font-weight: 500;
            cursor: pointer;
            transition: border-bottom 0.2s;
        }
        /* Style the selected page to look active */
        div.stRadio > label[data-baseweb="radio"] input:checked ~ div {
            border-bottom: 2px solid #1E90FF !important; /* Blue active tab indicator */
        }
        </style>
    """, unsafe_allow_html=True)
    
    current_page = st.radio("Navigation", PAGES, index=0, horizontal=True, label_visibility="collapsed")

    # Display the Main Title based on the page selection
    if current_page == "Applicant Prediction":
        st.title("💰 Loan Risk Advisor: Applicant Prediction")
    else:
        # Hide the main title on mock pages for a cleaner look
        st.markdown(f"<h1 style='visibility: hidden; height: 0;'>{current_page}</h1>", unsafe_allow_html=True)

    st.markdown("---") # Visual separator below the nav
    
    # --- ROUTING LOGIC ---
    model = load_model(MODEL_PATH)
    
    if current_page == "Applicant Prediction":
        render_prediction_page(model)
    elif current_page == "Insights":
        render_insights_page()
    elif current_page == "About":
        render_about_page()

# Run the main function
if __name__ == "__main__":
    main()
