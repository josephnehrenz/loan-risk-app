import streamlit as st
import datetime

# --- Constants ---
MODEL_DATE = datetime.date.today().strftime("%B %d, %Y")
KAGGLE_COMPETITION = "Kaggle Loan Payback Prediction Challenge" 
GITHUB_LINK = "https://github.com/yourusername/loan-risk-app" # <-- REMEMBER TO UPDATE THIS LINK!

# --- Page Function ---
def app():
    st.title("ℹ️ About This Loan Risk Model")
    st.markdown("---")
    
    st.header("Model Overview")
    st.info("This application hosts a predictive model designed to estimate the probability of a loan being paid back versus defaulting. It was developed using a complex XGBoost model trained on scaled and engineered features.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model & Training Details")
        st.markdown(f"""
        * **Algorithm:** **XGBoost Classifier** (Trained with Cross-Validation)
        * **Training Date:** **{MODEL_DATE}**
        * **Final AUC Score:** **~0.92** (from your log)
        * **Dataset Source:** {KAGGLE_COMPETITION}
        """)

    with col2:
        st.subheader("Data Access & Interpretabilty")
        st.markdown("""
        * **Feature Count:** **~30 Features** (Including engineered and encoded features).
        * **Input Data:** The model uses **standardized (scaled) inputs**.
        * **Interpretability:** Uses the **SHAP** library for clear, actionable explanations on every prediction.
        """)
        
    st.header("Source Code & Deployment")
    st.markdown(f"The source code and the trained model are hosted publicly:")
    st.code(GITHUB_LINK)
    st.link_button("View on GitHub", https://github.com/josephnehrenz/loan-risk-app/tree/main, help="Opens the project repository in a new tab.")

if __name__ == "__main__":
    app()
