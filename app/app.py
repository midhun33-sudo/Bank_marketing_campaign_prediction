import streamlit as st
import sys
import os

sys.path.append(
    os.path.abspath("../src")
)

from predict import predict_customer



st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦"
)

st.title(
    "🏦 Bank Marketing Campaign Prediction"
)

st.write(
    "Predict whether a customer will subscribe to a term deposit."
)



age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

education = st.selectbox(
    "Education",
    ["unknown","primary","secondary","tertiary"]
)

housing_loan = st.selectbox(
    "Housing Loan",
    ["yes","no"]
)

personal_loan = st.selectbox(
    "Personal Loan",
    ["yes","no"]
)

contacts_in_campaign = st.number_input(
    "Contacts In Campaign",
    min_value=0,
    value=1
)

contacted_in_before_campaing = st.number_input(
    "Previous Contacts",
    min_value=0,
    value=0
)

balance = st.number_input(
    "Balance",
    value=1000
)

last_call_duration = st.number_input(
    "Last Call Duration (seconds)",
    min_value=0,
    value=300
)

job = st.selectbox(
    "Job",
    [
        "blue-collar",
        "student",
        "management",
        "technician",
        "services",
        "retired",
        "entrepreneur",
        "housemaid",
        "unemployed",
        "self-employed",
        "unknown"
    ]
)

marital = st.selectbox(
    "Marital Status",
    [
        "married",
        "single",
        "divorced"
    ]
)

previous_outcome = st.selectbox(
    "Previous Outcome",
    [
        "unknown",
        "other",
        "success",
        "failure"
    ]
)

contact_type = st.selectbox(
    "Contact Type",
    [
        "telephone",
        "unknown",
        "cellular"
    ]
)


if st.button("Predict"):
    customer = {

        "age": age,

        "education": education,

        "housing_loan": housing_loan,

        "personal_loan": personal_loan,

        "contacts_in_campaign": contacts_in_campaign,

        "contacted_in_before_campaing":
            contacted_in_before_campaing,

        "balance": balance,

        "last_call_duration":
            last_call_duration,

        "job": job,

        "marital": marital,

        "previous_outcome":
            previous_outcome,

        "contact_type":
            contact_type
    }


    result = predict_customer(
        customer
    )


    st.subheader("Prediction Result")

    st.metric(
        "Subscription Probability",
        f"{result['probability']*100:.2f}%"
    )

    if result["prediction"] == "Will Subscribe":

        st.success(
            "✅ Customer is likely to Subscribe"
        )

    else:

        st.error(
            "❌ Customer is unlikely to Subscribe"
        )