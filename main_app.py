# Import libraries
import joblib
import streamlit as st

# Define functions and variables
with open('saved_models\model.joblib', 'rb') as model:
    classifier = joblib.load(model)

def predictor(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
       chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
       pH, sulphates, alcohol):
    global classifier
    prediction = classifier.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
       chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
       pH, sulphates, alcohol]])
    
    return int(prediction)

def main():
    # Title

    st.title('Wine Quality Prediction App')

    # Body
    fixed_acidity = st.number_input("Fixed Acidity")
    volatile_acidity =st.number_input("volatile_acidity")
    citric_acid =st.number_input("citric_acid")
    residual_sugar =st.number_input("residual_sugar")
    chlorides =st.number_input("chlorides")
    free_sulfur_dioxide =st.number_input("ree_sulfur_dioxide")
    total_sulfur_dioxide =st.number_input("Total_sulfur_dioxide")
    density =st.number_input("density ")
    pH =st.number_input("pH")
    sulphates =st.number_input("Sulphates")
    alcohol =st.number_input("Alcohol")


    # Predict
    if st.button('Predict'):
        prediction = predictor(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
       chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
       pH, sulphates, alcohol)
        st.success(f'The wine quality rating is {prediction}')
    
if __name__ == '__main__':
    main()
