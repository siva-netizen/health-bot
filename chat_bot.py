import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import csv

# Load data
training = pd.read_csv('Data/Training.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']

# Preprocess data
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

# Load additional data
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}
for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

# Function to calculate condition
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        if item in severityDictionary:
            sum = sum + severityDictionary[item]
        else:
            st.warning(f"Severity information not available for {item}")

    if (sum * days) / (len(exp) + 1) > 13:
        st.warning("You should take consultation from a doctor.")
    else:
        st.info("It might not be that bad, but you should take precautions.")

# Function to get description
def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

# Function to get severity dictionary
def getSeverityDict():
    global severityDictionary
    with open('MasterData/Symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

# Function to get precaution dictionary
def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

# Function to get user input for symptoms
def get_user_input():
    st.sidebar.header("HealthCare ChatBot")
    name = st.sidebar.text_input("Your Name?")
    st.sidebar.text(f"Hello, {name}")

    st.sidebar.text("Select a symptom from the menu:")
    selected_symptoms = st.sidebar.multiselect("Symptoms", x.columns)

    num_days = st.sidebar.number_input("From how many days?", min_value=0, step=1)
    return name, selected_symptoms, num_days

# Main function to run the app
def main():
    st.title("HealthCare ChatBot")

    name, selected_symptoms, num_days = get_user_input()

    if st.sidebar.button("Diagnose"):
        st.header("Diagnosis Result")

        # Check if valid symptoms are provided
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            related_symptoms = selected_symptoms.copy()
            
            # Ask about related symptoms
            while st.sidebar.checkbox("Do you have any related symptoms?"):
                new_symptom = st.sidebar.selectbox("Select a related symptom:", x.columns)
                related_symptoms.append(new_symptom)

            calc_condition(related_symptoms, num_days)

            st.subheader("Possible Conditions:")

            symptoms_input_df = pd.DataFrame(0, index=[0], columns=x.columns)
            for symptom in related_symptoms:
                if symptom in x.columns:
                    symptoms_input_df.loc[0, symptom] = 1
                else:
                    st.warning(f"Symptom '{symptom}' not recognized.")

            second_prediction = clf.predict(symptoms_input_df)
            present_disease = le.inverse_transform(second_prediction)
            st.write(f"You may have {present_disease[0]}")
            st.write(description_list[present_disease[0]])

            st.subheader("Precautions:")
            precaution_list = precautionDictionary[present_disease[0]]
            for i, j in enumerate(precaution_list):
                st.write(f"{i+1}. {j}")

if __name__ == '__main__':
    getSeverityDict()
    getDescription()
    getprecautionDict()
    main()
