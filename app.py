import streamlit as st
import pandas as pd
import pickle

st.title("Loan Approval AI-Powered App")
st.write("--------")
st.subheader("About the Model")

model_info = {
    "Model Name": "Random Forest Classifier",
    "Accuracy": 91,
    "Precision": 93,
    "Recall": 94
}

dat = pd.DataFrame(model_info, index=[1])
st.write(dat.head())

sidebar_title = st.sidebar.markdown("## Input Features")

ApplicantIncome = st.sidebar.number_input("Enter Applicant Income: ", min_value=2009.00, max_value=19988.00)
CoapplicantIncome = st.sidebar.number_input("Enter Coapplicant Income: ", min_value=1.00, max_value= 9996.00)
EmploymentStatus = st.sidebar.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed", "Contract", "Unemployed"]
)
Age = st.sidebar.number_input("Enter Age: ", min_value=21.00, max_value=59.00)
MaritalStatus = st.sidebar.selectbox(
    "Marital Status",
    ["Married", "Single"]
)
Dependents = st.sidebar.number_input("Enter Dependents: ", min_value=0.00, max_value=3.00)
CreditScore = st.sidebar.number_input("Enter Credit Score: ", min_value=550.00, max_value=799.00)
ExistingLoans = st.sidebar.number_input("Enter Existing Loans: ", min_value=0.00, max_value=4.00)
DTIRatio = st.sidebar.number_input("Enter DTI Ratio: ", min_value=0.10, max_value=0.60)
Savings = st.sidebar.number_input("Enter Savings: ", min_value=65.00, max_value=19996.00)
CollateralValue = st.sidebar.number_input("Enter Collateral Value: ", min_value=36.00, max_value=49954.00)
LoanAmount = st.sidebar.number_input("Enter LoanAmount: ", min_value=1015.00, max_value=39995.00)
LoanTerm = st.sidebar.number_input("Enter Loan Term: ", min_value=12.00, max_value=	84.00)
LoanPurpose = st.sidebar.selectbox(
    "Enter Loan Purpose",
    ["Personal", "Car", "Business", "Home", "Education"]
)
PropertyArea = st.sidebar.selectbox(
    "Enter Property Area",
    ["Urban", "Semiurban", "Rural"]
)
EducationLevel = st.sidebar.selectbox(
    "Enter Education Level:",
    ["Graduate", "Not Graduate"]
)	

Gender = st.sidebar.selectbox(
    "Enter Gender:",
    ["Male", "Female"]
)
EmployerCategory = st.sidebar.selectbox(
    "Enter Employer Category:",
    ["Business", "Private", "MNC", "Government", ""]
)

raw_data = pd.DataFrame({
    "Applicant_Income": [ApplicantIncome],
    "Coapplicant_Income": [CoapplicantIncome],
    "Employment_Status": [EmploymentStatus],
    "Age": [Age],
    "Marital_Status": [MaritalStatus],
    "Dependents": [Dependents],
    "Credit_Score": [CreditScore],
    "Existing_Loans": [ExistingLoans],
    "DTI_Ratio": [DTIRatio],
    "Savings": [Savings],
    "Collateral_Value": [CollateralValue],
    "Loan_Amount": [LoanAmount],
    "Loan_Term": [LoanTerm],
    "Loan_Purpose": [LoanPurpose],
    "Property_Area": [PropertyArea], 
    "Education_Level": [EducationLevel],
    "Gender": [Gender],
    "Employer_Category": [EmployerCategory]

})

st.write("------")
st.markdown("### Input Features")
st.write(raw_data.head())
encode_map = {
    'Employment_Status': {'Salaried': 1, 'Self-employed': 2, 'Contract': 3, 'Unemployed': 0},
    'Marital_Status':    {'Married': 1, 'Single': 0},
    'Loan_Purpose':      {'Personal': 1, 'Car': 2, 'Business': 3, 'Home': 4, 'Education': 5},
    'Property_Area':     {'Urban': 1, 'Semiurban': 2, 'Rural': 0},
    'Education_Level':   {'Not Graduate': 0, 'Graduate': 1},
    'Gender':            {'Male': 1, 'Female': 0},
    'Employer_Category': {'Private': 2, 'Unemployed': 0, 'Government': 4, 'MNC': 3, 'Business': 1},
}


for col, mapping in encode_map.items():
    raw_data[col] = raw_data[col].map(mapping)

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.write("-----")
st.markdown("### Predict Loan")
if st.button("Predict"):
    prediction = model.predict(raw_data)
    
    if prediction == 1:
        st.write("Congratulations! Your Loan has been APPROVED")

    else:
        st.write("Loan NOT APPROVED")