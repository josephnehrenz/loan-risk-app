# streamlit_app.py
# This file is the required entry point for Streamlit deployment.
# It imports and runs the main function from your actual primary page 
# located in the 'pages' subdirectory.

from pages.p1_Loan_Predictor import main

if __name__ == "__main__":
    main()
