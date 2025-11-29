# Loan Risk Advisor

An interactive web application that predicts loan repayment probability and provides model insights using machine learning. Built with XGBoost and Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loan-repayment-risk-app.streamlit.app/)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/josephnehrenz/s5e11-predicting-loan-payback-with-xgb)

## About the Application

The Loan Risk Advisor helps financial institutions and loan officers assess applicant risk through two main components:

** Applicant Prediction Page:**
- Input applicant financial and demographic information
- Get real-time repayment probability scores
- View SHAP waterfall plots explaining feature contributions
- Compare against global average repayment rates

** Model Insights Page:**
- Explore global feature importance using SHAP analysis
- Understand key drivers of loan repayment decisions
- View feature distributions and model behavior patterns

## Technical Details

### Machine Learning
- **Model**: XGBoost Classifier
- **Features**: 20 engineered features including financial ratios, credit metrics, and target-encoded categorical variables
- **Target**: Probability of loan repayment (binary classification)
- **Key Features**: Credit score, annual income, debt-to-income ratio, employment status, loan grade, and more

### Web Application
- **Framework**: Streamlit
- **Visualization**: SHAP, Matplotlib
- **Interactive Elements**: Real-time predictions, dynamic sliders, explanatory AI insights

## Model Features

The model uses 20 carefully engineered features:

### Financial Metrics (13 features)
- `annual_income`, `credit_score`, `loan_amount`, `interest_rate`
- `debt_to_income_ratio`, `income_loan_ratio`, `loan_to_income`
- `total_debt`, `available_income`, `monthly_payment_approx`
- `payment_to_income`, `default_risk_score`, `grade_number`

### Target-Encoded Categorical Features (7 features)
- `TE_gender`, `TE_marital_status`, `TE_education_level`
- `TE_employment_status`, `TE_loan_purpose`
- `TE_grade_subgrade`, `TE_grade_letter`

## Model Performance

- **Global Baseline Repayment Rate**: 79.9%
- **Interpretability**: Full SHAP-based explainability for each prediction
- **Key Insights**: Employment status and credit quality are the strongest predictors of repayment

## Live Demo

Try the live app here:  
**[https://your-app-name.streamlit.app](https://loan-repayment-risk-app.streamlit.app/)**

## Project Structure

```
loan-risk-advisor/
├── app.py              # Main prediction application
├── insights.py         # Model insights dashboard
├── xgboost_model.json  # Trained XGBoost model
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation & Local Development

To run this application locally:

```bash
# Clone the repository
git clone https://github.com/your-username/loan-risk-advisor.git
cd loan-risk-advisor

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app (Prediction Page)
streamlit run app.py

# Or run the Insights Page directly
streamlit run insights.py
```

## Key Features & Insights
### Top Model Drivers
1. Employment Status - Most significant risk split (Employed/Retired vs Unemployed)
2. Credit Quality - Credit score and loan grade are primary indicators
3. Financial Capacity - Income levels and debt ratios determine repayment ability
4. Loan Terms - Amount, interest rate, and payment burdens

### Business Value
1. Transparent AI: Every prediction comes with detailed explanations
2. Risk Assessment: Quantifiable repayment probabilities for informed decisions
3. Regulatory Compliance: Model interpretability supports responsible lending practices

## Contributing

Feel free to fork this project and submit pull requests for any improvements!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built for the Kaggle Playground Series - Season 5, Episode 11*
