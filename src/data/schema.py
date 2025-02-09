from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel

SYN_Q_PROMPT = (
    "Here are five entries of tabular data in JSON format, each consisting of {} features. "
    "Each feature is described in a structured JSON format: \"feature name\": \"value\" . "
    "The target feature {} is a {} task."
    "\n\n{}"
    "Directly generate only one new final sample in JSON format that approximates the key patterns observed in the provided samples. \n"
    "Answer:\n"
)

SYN_A_PROMPT = "Here's a new JSON format sample that attempts to approximate the key patterns observed in the provided samples:\n{}"

# LLM_Q_PROMPT_DICT = {
#     "abalone": (
#         "Predict the age of abalone from physical measurements showed in Text.\n"Text: {}. "
#         "Directly respond with an integer. \n"
#         "\nAnswer: "
#     ),

#     "adult": (
#         "Determine whether a person makes over $50K a year based on personal attributes show in Text.\n"Text: {}. "
#         "Directly respond with only either 'yes' or 'no'. \n"
#         "\nAnswer: "
#     ),

#     "buddy": (
#         "Detect the breed of an animal based on its condition, appearance, and other factors show in Text.\n"Text: {}. "
#         "Directly respond with only either 'A', 'B', or 'C'. \n"
#         "\nAnswer: "
#     ),

#     "california": (
#         "Predict the median house price in California based on features such as population count, "
#         "median income, median house age and etc show in Text.\n"Text: {}. "
#         "Directly respond with a floating-point value formatted to 4 decimal places. \n"
#         "\nAnswer: "
#     ),

#     "diabetes": (
#         "Predict whether a patient has diabetes based on certain diagnostic measurements show in Text.\n"Text: {}. "
#         "Directly respond with only either 'negative' or 'positive'. \n"
#         "\nAnswer: "
#     ),

#     "insurance": (
#         "Predict the insurance charges for a person based on their age, sex, body mass index (BMI), "
#         "number of children, smoking status, and region show in the Text, \n"Text: {}. "
#         "Directly respond with an integer. \n"
#         "\nAnswer: "
#     ),
# }

LLM_Q_PROMPT_DICT = {
    "abalone": (
        "Predict the age of the abalone using the physical measurements provided in the text below. "
        "Directly respond with an integer. "
        "\nText: {}. "
        "\nAnswer: "
    ),

    "adult": (
        "Determine whether a person earns over $50K per year based on the personal attributes provided in the text below. "
        "Directly respond with only either 'yes' or 'no'. "
        "\nText: {}. "
        "\nAnswer: "
    ),

    "buddy": (
        "Identify the breed of an animal based on its condition, appearance, and other factors described in the text below. "
        "Directly respond with only either 'A', 'B', or 'C'. "
        "\nText: {}. "
        "\nAnswer: "
    ),

    "california": (
        "Predict the median house price in California using features such as population count, median income, median house age, and other details provided in the text below. "
        "Directly respond with a floating-point value formatted to 2 decimal places. "
        "\nText: {}. "
        "\nAnswer: "
    ),

    "diabetes": (
        "Predict whether a patient has diabetes based on the diagnostic measurements provided in the text below. "
        "Directly respond with only either 'negative' or 'positive'. "
        "\nText: {}. "
        "\nAnswer: "
    ),

    "insurance": (
        "Predict the insurance charges for a person using their age, sex, body mass index (BMI), number of children, smoking status, and region as described in the text below. "
        "Directly respond with an integer. "
        "\nText: {}. "
        "\nAnswer: "
    ),
    "german": (
        "Infer the creditworthiness of a customer based on the following financial profile. "
        "Respond with 'good' or 'bad' only. "
        "For example, a customer with stable income, no debts, and property ownership should be classified as 'good'. "
        "The customer's profile consists of the following attribute-value pairs: "
        "\n{}. "
        "\nAnswer: "
    ),

    "lendingclub": (
        "Assess the loan status of the client based on the loan records provided by Lending Club. "
        "Respond with 'good' or 'bad' only. "
        "For example, a client with stable income, no debts, and property ownership should be classified as 'good'. "
        "The client's profile consists of the following attribute-value pairs: "
        "\n{}. "
        "\nAnswer: "
    ),
    "travel": (
        "Infer the claim status of the travel insurance company based on the provided information. "
        "Your response should be either 'yes' or 'no' only. "
        "For instance, the insurance company has attributes: 'Name of Agency: cbh, Type of Travel Insurance Agencies: travel agency, Distribution Channel of Travel Insurance Agencies: offline, Name of the Travel Insurance Products: comprehensive plan, Duration of Travel: 186, Destination of Travel: malaysia, Amount of Sales of Travel Insurance Policies: -29, Commission Received for Travel Insurance Agency: 9.57, Age of Insured: 81'. the claim status should be inferred as 'no'"
        "\nThe insurance company's profile consists of the following attribute-value pairs: "
        "\n{}. "
        "\nAnswer: "
    ),
    
    "cleveland": (
        "Predict whether the patient has heart disease using the features below: Age, Sex, "
        "ChestPainType, RestingBloodPressure, Cholesterol, FastingBloodSugar, RestingECG, "
        "MaxHeartRate, ExerciseInducedAngina, OldPeak, PeakExerciseSTSegmentSlope, "
        "NumberOfVesselsColored, and ThalassemiaStatus. "
        "Directly respond with 'unhealthy' if the patient has heart disease or 'healthy' if they do not. "
        "\nText: {}. "
        "\nAnswer: "
    ),
    "hungarian": (
        "Predict whether the patient has heart disease using the features below: Age, Sex, "
        "ChestPainType, RestingBloodPressure, Cholesterol, FastingBloodSugar, RestingECG, "
        "MaxHeartRate, ExerciseInducedAngina, OldPeak, PeakExerciseSTSegmentSlope, "
        "NumberOfVesselsColored, and ThalassemiaStatus. "
        "Directly respond with 'unhealthy' if the patient has heart disease or 'healthy' if they do not. "
        "\nText: {}. "
        "\nAnswer: "
    ),
    "switzerland": (
        "Predict whether the patient has heart disease using the features below: Age, Sex, "
        "ChestPainType, RestingBloodPressure, Cholesterol, FastingBloodSugar, RestingECG, "
        "MaxHeartRate, ExerciseInducedAngina, OldPeak, PeakExerciseSTSegmentSlope, "
        "NumberOfVesselsColored, and ThalassemiaStatus. "
        "Directly respond with 'unhealthy' if the patient has heart disease or 'healthy' if they do not. "
        "\nText: {}. "
        "\nAnswer: "
    ),
    "va": (
        "Predict whether the patient has heart disease using the features below: Age, Sex, "
        "ChestPainType, RestingBloodPressure, Cholesterol, FastingBloodSugar, RestingECG, "
        "MaxHeartRate, ExerciseInducedAngina, OldPeak, PeakExerciseSTSegmentSlope, "
        "NumberOfVesselsColored, and ThalassemiaStatus. "
        "Directly respond with 'unhealthy' if the patient has heart disease or 'healthy' if they do not. "
        "\nText: {}. "
        "\nAnswer: "
    ),
    
    # "travel": (
    #     "Infer the claim status of the travel insurance company based on the provided information. "
    #     "Your response should be either 'approved' or 'rejected' only. "
    #     "For instance: \'The insurance company has attributes: Agency: CBH, Agency Type: Travel Agency, Distribution Chanel: Offline, Product Name: Comprehensive Plan, Duration: 186, Destination: MALAYSIA, Net Sales: -29, Commision: 9.57, Age: 81.\', should be classified as rejected. "
    #     "\nThe insurance company's profile consists of the following attribute-value pairs: "
    #     "\n{}. "
    #     "\nClaim Status: ? "
    #     "\nAnswer: "
    # ),
    # "travel": (
    #     "Evaluate the claim status of the insurance company based on the following attributes. "
    #     "Respond with only 'valid' or 'invalid'. "
    # "The attributes include 5 categorical and 4 numerical features, defined as follows: "
    # "- Agency: The name of the travel insurance agency (categorical). "
    # "- Agency Type: The type of travel insurance agency (categorical). "
    # "- Distribution Channel: The method of distribution for travel insurance (categorical). "
    # "- Product Name: The name of the travel insurance product (categorical). "
    # "- Duration: The duration of the travel (categorical). "
    # "- Destination: The travel destination (numerical). "
    # "- Net Sales: The sales amount for travel insurance policies (numerical). "
    # "- Commission: The commission earned by the travel insurance agency (numerical). "
    # "- Age: The age of the insured individual (numerical). "
    #     "The insurance company details are provided in the following JSON format: "
    #     "\nJSON: \n{}. "
    #     "\nAnswer: "
    # )
}

LLM_A_PROMPT = "{}"


def map_abalone(x):
    # Regression
    return str(int(x))


def map_adult(x):
    # Classification
    return {1: "yes", 0: "no"}.get(x, "unknown")


def map_buddy(x):
    # Classification
    return {0: "A", 1: "B", 2: "C"}.get(x, "unknown")


def map_california(x):
    # Regression
    return f"{float(x):.2f}"


def map_diabetes(x):
    # Classification
    return {0: "negative", 1: "positive"}.get(x, "unknown")


def map_insurance(x):
    # Regression
    return str(int(x))


def map_german(x):
    # Regression
    return {"1.0": "good", "2.0": "bad"}.get(x, "unknown")


def map_lendingclub(x):
    x = x.strip()
    return {"fullypaid": "good", "chargedoff": "bad"}.get(x, "unknown")


def map_travel(x):
    x = x.strip()
    return {"yes": "yes", "no": "no"}.get(x, "unknown")

def map_cleveland(x):
    return {'healthy':'healthy', 'heartdisease':'unhealthy'}.get(x, "unknown")

def map_hungarian(x):
    return {'healthy':'healthy', 'heartdisease':'unhealthy'}.get(x, "unknown")

def map_switzerland(x):
    return {'healthy':'healthy', 'heartdisease':'unhealthy'}.get(x, "unknown")

def map_va(x):
    return {'healthy':'healthy', 'heartdisease':'unhealthy'}.get(x, "unknown")


TARGET_MAP_DICT = {
    "abalone": map_abalone,
    "adult": map_adult,
    "buddy": map_buddy,
    "california": map_california,
    "diabetes": map_diabetes,
    "insurance": map_insurance,
    "german": map_german,
    "travel": map_travel,
    "lendingclub": map_lendingclub,
    "cleveland": map_cleveland,
    "hungarian": map_hungarian,
    "switzerland": map_switzerland,
    "va": map_va
}

INVERSE_TARGET_MAP_DICT = {
    "abalone": int,  # Regression
    "adult": {"yes": 1, "no": 0},  # Classification
    "buddy": {"A": 0, "B": 1, "C": 2},  # Classification
    "california": float,  # Regression
    "diabetes": {"negative": 0, "positive": 1},  # Classification
    "insurance": int,  # Regression
    "german": {"good": "1.0", "bad": "2.0"},  # Classification
    "lendingclub": {"good": "fullypaid", "bad": "chargedoff"},  # Classification
    "travel": {"yes": "yes", "no": "no"},  # Classification
    "cleveland": {"healthy": "healthy", "unhealthy": "heartdisease"},
    "hungarian": {"healthy": "healthy", "unhealthy": "heartdisease"},
    "switzerland":{"healthy": "healthy", "unhealthy": "heartdisease"},
    "va": {"healthy": "healthy", "unhealthy": "heartdisease"},
}


class BaseDataSchema(BaseModel):
    dtype_dict: Dict[str, Any]
    target_type: str
    target: str

    syn_q_prompt: str
    syn_a_prompt: str

    llm_q_prompt: str
    llm_a_prompt: str

    target_map: Callable
    inverse_target_map: object

    class Config:
        extra = "forbid"
        frozen = True

    @staticmethod
    def dict_to_jsonstr(d) -> str:

        # Apply formatting to dictionary values
        formatted_dict = {key: BaseDataSchema.format_value(val) for key, val in d.items()}

        # Manually build the JSON-like string with indentation
        content = ",\n\t".join(f'"{key}": {val}' for key, val in formatted_dict.items())
        return f"{{\n\t{content}\n}}"

    @staticmethod
    def dict_to_str(d: Dict[str, Any]) -> str:
        s_tok = "'"
        e_tok = "'"
        sep = ", "
        # template = "the state of {} is {}"
        template = "{}: {}"

        content = [template.format(key, BaseDataSchema.format_value(val)) for key, val in d.items()]

        s = f"{s_tok} {sep.join(content)} {e_tok}"
        s = s.replace("nan", "norecord")
        return s 

    @staticmethod
    def format_value(val: Any) -> str:
        if isinstance(val, float):
            return f'{val:.4f}'
        elif isinstance(val, str):
            return f'{val.lower()}'
        else:
            return f'{val}'

    def format_load_csv(self, csv_path):

        data = pd.read_csv(csv_path, dtype=self.dtype_dict)
        # data = data.dropna(axis=0, how="any")
        data = data.map(lambda x: x.lower() if isinstance(x, str) else x)
        data = data.reset_index(drop=True)
        data = data.to_dict(orient="records")

        return data


ABALONE_SCHEMA = BaseDataSchema(

    dtype_dict={
        "Sex": np.str_,
        "Length": np.float64,
        "Diameter": np.float64,
        "Height": np.float64,
        "Whole_weight": np.float64,
        "Shucked_weight": np.float64,
        "Viscera_weight": np.float64,
        "Shell_weight": np.float64,
        "Class_number_of_rings": np.int64,
    },
    target_type="regression",
    target="Class_number_of_rings",
    syn_q_prompt=SYN_Q_PROMPT.format(
        "nine", "Class_number_of_rings", "regression", "{}"),
    syn_a_prompt=SYN_A_PROMPT,
    llm_q_prompt=LLM_Q_PROMPT_DICT["abalone"],
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["abalone"],
    inverse_target_map=INVERSE_TARGET_MAP_DICT["abalone"],
)

ADULT_SCHEMA = BaseDataSchema(


    dtype_dict={
        "age": np.float64,
        "workclass": np.str_,
        "fnlwgt": np.float64,
        "education": np.str_,
        "education-num": np.float64,
        "marital-status": np.str_,
        "occupation": np.str_,
        "relationship": np.str_,
        "race": np.str_,
        "sex": np.str_,
        "capital-gain": np.float64,
        "capital-loss": np.float64,
        "hours-per-week": np.float64,
        "native-country": np.str_,
        "class": np.int64,
    },
    target_type="classification",
    target="class",
    syn_q_prompt=SYN_Q_PROMPT.format(
        "fifteen", "class", "classification", "{}"),
    syn_a_prompt=SYN_A_PROMPT,
    llm_q_prompt=LLM_Q_PROMPT_DICT["adult"],
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["adult"],
    inverse_target_map=INVERSE_TARGET_MAP_DICT["adult"],

)

BUDDY_SCHEMA = BaseDataSchema(


    dtype_dict={
        "issue_date": np.float64,
        "listing_date": np.float64,
        "condition": np.float64,
        "color_type": np.str_,
        "length(m)": np.float64,
        "height(cm)": np.float64,
        "X1": np.int64,
        "X2": np.int64,
        "pet_category": np.int64,
        "breed_category": np.int64,
    },
    target_type="classification",
    target="breed_category",
    syn_q_prompt=SYN_Q_PROMPT.format(
        "ten", "breed_category", "classification", "{}"),
    syn_a_prompt=SYN_A_PROMPT,
    llm_q_prompt=LLM_Q_PROMPT_DICT["buddy"],
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["buddy"],
    inverse_target_map=INVERSE_TARGET_MAP_DICT["buddy"],
)

CALIFORNIA_SCHEMA = BaseDataSchema(


    dtype_dict={
        "longitude": np.float64,
        "latitude": np.float64,
        "housing_median_age": np.float64,
        "total_rooms": np.float64,
        "total_bedrooms": np.float64,
        "population": np.float64,
        "households": np.float64,
        "median_income": np.float64,
        "median_house_value": np.float64,
    },
    target_type="regression",
    target="median_house_value",
    syn_q_prompt=SYN_Q_PROMPT.format(
        "nine", "median_house_value", "regression", "{}"),
    syn_a_prompt=SYN_A_PROMPT,
    llm_q_prompt=LLM_Q_PROMPT_DICT["california"],
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["california"],
    inverse_target_map=INVERSE_TARGET_MAP_DICT["california"],

)

DIABETES_SCHEMA = BaseDataSchema(


    dtype_dict={
        "Pregnancies": np.int64,
        "Glucose": np.float64,
        "BloodPressure": np.float64,
        "SkinThickness": np.float64,
        "Insulin": np.float64,
        "BMI": np.float64,
        "DiabetesPedigreeFunction": np.float64,
        "Age": np.float64,
        "Outcome": np.int64,
    },
    target_type="classification",
    target="Outcome",
    syn_q_prompt=SYN_Q_PROMPT.format(
        "nine", "Outcome", "classification", "{}"),
    syn_a_prompt=SYN_A_PROMPT,
    llm_q_prompt=LLM_Q_PROMPT_DICT["diabetes"],
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["diabetes"],
    inverse_target_map=INVERSE_TARGET_MAP_DICT["diabetes"],


)

INSURANCE_SCHEMA = BaseDataSchema(

    dtype_dict={
        "age": np.int64,
        "sex": np.str_,
        "bmi": np.float64,
        "children": np.int64,
        "smoker": np.str_,
        "region": np.str_,
        "charges": np.int64,
    },
    target_type="regression",
    target="charges",
    syn_q_prompt=SYN_Q_PROMPT.format(
        "seven", "charges", "regression", "{}"),
    syn_a_prompt=SYN_A_PROMPT,
    llm_q_prompt=LLM_Q_PROMPT_DICT["insurance"],
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["insurance"],
    inverse_target_map=INVERSE_TARGET_MAP_DICT["insurance"],
)


GERMAN_SCHEMA = BaseDataSchema(
    dtype_dict={
        "Status_of_existing_checking_account": np.str_,
        "Duration_in_month": np.str_,
        "Credit_history": np.str_,
        "Purpose": np.str_,
        "Credit_amount": np.float64,
        "Savings_account_or_bonds": np.str_,
        "Present_employment_since": np.str_,
        "Installment_rate_in_percentage_of_disposable_income": np.str_,
        "Personal_status_and_sex": np.str_,
        "Other_debtors_or_guarantors": np.str_,
        "Present_residence_since": np.str_,
        "Property": np.str_,
        "Age_in_years": np.int64,
        "Other_installment_plans": np.str_,
        "Housing": np.str_,
        "Number_of_existing_credits_at_this_bank": np.str_,
        "Job": np.str_,
        "Number_of_people_being_liable_to_provide_maintenance_for": np.str_,
        "Telephone": np.str_,
        "Foreign_worker": np.str_,
        "Status": np.str_
    },
    target_type="classification",
    target="Status",
    syn_q_prompt=SYN_Q_PROMPT.format(
        "twenty-one", "Status", "classification", "{}"),
    syn_a_prompt=SYN_A_PROMPT,
    llm_q_prompt=LLM_Q_PROMPT_DICT["german"],
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["german"],
    inverse_target_map=INVERSE_TARGET_MAP_DICT["german"],

)

TRAVEL_SCHEMA = BaseDataSchema(
    dtype_dict = {
        "Name_of_Agency": np.str_,
        "Type_of_Travel_Insurance_Agencies": np.str_,
        "Distribution_Channel_of_Travel_Insurance_Agencies": np.str_,
        "Name_of_the_Travel_Insurance_Products": np.str_,
        "Claim_Status": np.str_,
        "Duration_of_Travel": np.str_,
        "Destination_of_Travel": np.str_,
        "Amount_of_Sales_of_Travel_Insurance_Policies": np.float64,
        "Commission_Received_for_Travel_Insurance_Agency": np.float64,
        "Age_of_Insured": np.int64,
    },
    target_type="classification",
    target="Claim_Status",
    syn_q_prompt=SYN_Q_PROMPT.format(
        "ten", "Claim_Status", "classification", "{}"),
    syn_a_prompt=SYN_A_PROMPT,
    llm_q_prompt=LLM_Q_PROMPT_DICT["travel"],
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["travel"],
    inverse_target_map=INVERSE_TARGET_MAP_DICT["travel"],
)

LENDINGCLUB_SCHEMA = BaseDataSchema(
    dtype_dict = {
        "Installment": np.float64,
        "Loan_Purpose": np.str_,
        "Loan_Application_Type": np.str_,
        "Interest_Rate": np.float64,
        "Last_Payment_Amount": np.float64,
        "Loan_Amount": np.float64,
        "Revolving_Balance": np.float64,
        "Delinquency_In_2_years": np.float64,
        "Inquiries_In_6_Months": np.float64,
        "Mortgage_Accounts": np.float64,
        "Grade": np.str_,
        "Open_Accounts": np.float64,
        "Revolving_Utilization_Rate": np.float64,
        "Total_Accounts": np.float64,
        "Fico_Range_Low": np.float64,
        "Fico_Range_High": np.float64,
        "Address_State": np.str_,
        "Employment_Length": np.str_,
        "Home_Ownership": np.str_,
        "Verification_Status": np.str_,
        "Annual_Income": np.float64,
        "Loan_Status": np.str_,
    },
    target_type="classification",
    target="Loan_Status",
    syn_q_prompt=SYN_Q_PROMPT.format(
        "twenty-two", "Loan_Status", "classification", "{}"),
    syn_a_prompt=SYN_A_PROMPT,
    llm_q_prompt=LLM_Q_PROMPT_DICT["lendingclub"],
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["lendingclub"],
    inverse_target_map=INVERSE_TARGET_MAP_DICT["lendingclub"],
)


CLEVELAND_SCHEMA = BaseDataSchema(
    dtype_dict={
        "Age": np.float64,
        "Sex": np.str_,
        "ChestPainType": np.str_,
        "RestingBloodPressure": np.float64,
        "Cholesterol": np.float64,
        "FastingBloodSugar": np.str_,
        "RestingECG": np.str_,
        "MaxHeartRate": np.float64,
        "ExerciseInducedAngina": np.str_,
        "OldPeak": np.float64,
        "PeakExerciseSTSegmentSlope": np.str_,
        "NumberOfVesselsColored": np.float64,
        "ThalassemiaStatus": np.str_,
        "heartdiseaseStatus": np.str_,  
    },
    target_type="classification",
    target="heartdiseaseStatus",
    # This assumes you have a SYN_Q_PROMPT that takes 4 format arguments:
    # (number_of_columns, target_column, "classification", "{}")
    # If your prompt takes different arguments, adjust accordingly.
    syn_q_prompt=SYN_Q_PROMPT.format(
        "fourteen",               # Because we have 14 total columns
        "heartdiseaseStatus",     # Your target column name
        "classification",         # Your target type
        "{}"                      # Placeholder for data row
    ),
    syn_a_prompt=SYN_A_PROMPT,    
    llm_q_prompt=LLM_Q_PROMPT_DICT["cleveland"],   # You’ll need an entry for "heart" in LLM_Q_PROMPT_DICT
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["cleveland"],       # Map from raw values (e.g., 0/1) to model-friendly labels
    inverse_target_map=INVERSE_TARGET_MAP_DICT["cleveland"],  # Reverse mapping (for predictions)
)

HUNGARIAN_SCHEMA = BaseDataSchema(
    dtype_dict={
        "Age": np.float64,
        "Sex": np.str_,
        "ChestPainType": np.str_,
        "RestingBloodPressure": np.float64,
        "Cholesterol": np.float64,
        "FastingBloodSugar": np.str_,
        "RestingECG": np.str_,
        "MaxHeartRate": np.float64,
        "ExerciseInducedAngina": np.str_,
        "OldPeak": np.float64,
        "PeakExerciseSTSegmentSlope": np.str_,
        "NumberOfVesselsColored": np.float64,
        "ThalassemiaStatus": np.str_,
        "heartdiseaseStatus": np.str_,  
    },
    target_type="classification",
    target="heartdiseaseStatus",
    # This assumes you have a SYN_Q_PROMPT that takes 4 format arguments:
    # (number_of_columns, target_column, "classification", "{}")
    # If your prompt takes different arguments, adjust accordingly.
    syn_q_prompt=SYN_Q_PROMPT.format(
        "fourteen",               # Because we have 14 total columns
        "heartdiseaseStatus",     # Your target column name
        "classification",         # Your target type
        "{}"                      # Placeholder for data row
    ),
    syn_a_prompt=SYN_A_PROMPT,    
    llm_q_prompt=LLM_Q_PROMPT_DICT["hungarian"],   # You’ll need an entry for "heart" in LLM_Q_PROMPT_DICT
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["hungarian"],       # Map from raw values (e.g., 0/1) to model-friendly labels
    inverse_target_map=INVERSE_TARGET_MAP_DICT["hungarian"],  # Reverse mapping (for predictions)
)


SWITZERLAND_SCHEMA = BaseDataSchema(
    dtype_dict={
        "Age": np.float64,
        "Sex": np.str_,
        "ChestPainType": np.str_,
        "RestingBloodPressure": np.float64,
        "Cholesterol": np.float64,
        "FastingBloodSugar": np.str_,
        "RestingECG": np.str_,
        "MaxHeartRate": np.float64,
        "ExerciseInducedAngina": np.str_,
        "OldPeak": np.float64,
        "PeakExerciseSTSegmentSlope": np.str_,
        "NumberOfVesselsColored": np.float64,
        "ThalassemiaStatus": np.str_,
        "heartdiseaseStatus": np.str_,  
    },
    target_type="classification",
    target="heartdiseaseStatus",
    # This assumes you have a SYN_Q_PROMPT that takes 4 format arguments:
    # (number_of_columns, target_column, "classification", "{}")
    # If your prompt takes different arguments, adjust accordingly.
    syn_q_prompt=SYN_Q_PROMPT.format(
        "fourteen",               # Because we have 14 total columns
        "heartdiseaseStatus",     # Your target column name
        "classification",         # Your target type
        "{}"                      # Placeholder for data row
    ),
    syn_a_prompt=SYN_A_PROMPT,    
    llm_q_prompt=LLM_Q_PROMPT_DICT["switzerland"],   # You’ll need an entry for "heart" in LLM_Q_PROMPT_DICT
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["switzerland"],       # Map from raw values (e.g., 0/1) to model-friendly labels
    inverse_target_map=INVERSE_TARGET_MAP_DICT["switzerland"],  # Reverse mapping (for predictions)
)




VA_SCHEMA = BaseDataSchema(
    dtype_dict={
        "Age": np.float64,
        "Sex": np.str_,
        "ChestPainType": np.str_,
        "RestingBloodPressure": np.float64,
        "Cholesterol": np.float64,
        "FastingBloodSugar": np.str_,
        "RestingECG": np.str_,
        "MaxHeartRate": np.float64,
        "ExerciseInducedAngina": np.str_,
        "OldPeak": np.float64,
        "PeakExerciseSTSegmentSlope": np.str_,
        "NumberOfVesselsColored": np.float64,
        "ThalassemiaStatus": np.str_,
        "heartdiseaseStatus": np.str_,  
    },
    target_type="classification",
    target="heartdiseaseStatus",
    # This assumes you have a SYN_Q_PROMPT that takes 4 format arguments:
    # (number_of_columns, target_column, "classification", "{}")
    # If your prompt takes different arguments, adjust accordingly.
    syn_q_prompt=SYN_Q_PROMPT.format(
        "fourteen",               # Because we have 14 total columns
        "heartdiseaseStatus",     # Your target column name
        "classification",         # Your target type
        "{}"                      # Placeholder for data row
    ),
    syn_a_prompt=SYN_A_PROMPT,    
    llm_q_prompt=LLM_Q_PROMPT_DICT["va"],   # You’ll need an entry for "heart" in LLM_Q_PROMPT_DICT
    llm_a_prompt=LLM_A_PROMPT,
    target_map=TARGET_MAP_DICT["va"],       # Map from raw values (e.g., 0/1) to model-friendly labels
    inverse_target_map=INVERSE_TARGET_MAP_DICT["va"],  # Reverse mapping (for predictions)
)



SCHEMA_MAP = {
    "abalone": ABALONE_SCHEMA,
    "adult": ADULT_SCHEMA,
    "buddy": BUDDY_SCHEMA,
    "california": CALIFORNIA_SCHEMA,
    "diabetes": DIABETES_SCHEMA,
    "insurance": INSURANCE_SCHEMA,
    "german": GERMAN_SCHEMA,
    "travel": TRAVEL_SCHEMA,
    "lendingclub": LENDINGCLUB_SCHEMA,
    "cleveland": CLEVELAND_SCHEMA,
    "hungarian": HUNGARIAN_SCHEMA,
    "switzerland": SWITZERLAND_SCHEMA,
    "va": VA_SCHEMA,
    "switzerland_healthy":SWITZERLAND_SCHEMA,
    "va_healthy":VA_SCHEMA
}


if __name__ == "__main__":
    for k, v in SCHEMA_MAP.items():
        print(f"Class Name: {k}")
        # Iterate over the attributes and their values
        for field_name, field_info in v.model_fields.items():
            value = getattr(v, field_name, None)
            print(f"  {field_name}: {value}")
        print("\n\n")
