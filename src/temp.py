from predict import predict_customer




test_customer = {

    "age": 35,
    "education": "secondary",
    "housing_loan": "yes",
    "personal_loan": "no",
    "contacts_in_campaign": 2,
    "contacted_in_before_campaing": 1,
    "balance": 1500,
    "last_call_duration": 300,
    "job": "blue-collar",
    "marital": "married",
    "previous_outcome": "success",
    "contact_type": "telephone"
}

result = predict_customer(
    test_customer
)

print(result)


high_subscriber = {

    "age": 55,

    "education": "tertiary",

    "housing_loan": "no",

    "personal_loan": "no",

    "contacts_in_campaign": 1,

    "contacted_in_before_campaing": 3,

    "balance": 25000,

    "last_call_duration": 1200,

    "job": "student",

    "marital": "single",

    "previous_outcome": "success",

    "contact_type": "telephone"
}

print(
    predict_customer(
        high_subscriber
    )
)