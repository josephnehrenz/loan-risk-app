import streamlit as st

# Set the overall page configuration (title, layout, etc.)
st.set_page_config(
    page_title="Loan Risk Advisor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# This is the content for the default Home page (the one that Streamlit auto-selects).
# It serves as a navigational landing page, letting the files in the 'pages/' folder 
# populate the sidebar correctly.

st.title("Welcome to the Loan Risk Advisor")
st.markdown("Please use the sidebar on the left to navigate to the **Applicant Prediction** or **Model Insights** pages.")
