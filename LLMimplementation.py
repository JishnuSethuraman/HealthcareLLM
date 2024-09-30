import joblib
import numpy as np
import os
import openai

# Set up OpenAI API key for LLM
openai.api_key = 'yourapikey'

# Load the trained TabNet model
model = joblib.load('tabnet_healthcare_model.pkl')

# Load the encoders and scaler
encoders = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Function to encode the user input
def encode_input(user_input):
    """
    Encode the user input using the saved encoders and scaler.
    """
    gender = encoders['Gender'].transform([user_input['gender'].capitalize()])[0]
    blood_type = encoders['Blood Type'].transform([user_input['blood_type']])[0]
    condition = encoders['Medical Condition'].transform([user_input['medical_condition'].capitalize()])[0]
    
    age = user_input['age']
    stay_period = user_input['stay_period']
    
    X_input = np.array([[age, stay_period, gender, blood_type, condition]])
    
    X_input[:, :2] = scaler.transform(X_input[:, :2])  # Scale the numerical features
    return X_input

# Function to predict healthcare cost using TabNet
def predict_healthcare_cost(user_input):
    X_input = encode_input(user_input)
    prediction = model.predict(X_input)
    predicted_cost = np.expm1(prediction)  # Reverse log transformation
    return predicted_cost[0]

# Function to interact with LLM
def chat_with_llm():
    prompt = """
    You are a virtual assistant trained to estimate healthcare costs.
    Please ask the user the following questions to gather the necessary information:
    1. What is your age?
    2. What is your gender? (male/female)
    3. What is your blood type? (A, B, O, AB)
    4. What medical condition are you experiencing? (e.g., cold, flu, chest pain, pneumonia, cancer)
    5. How long do you expect to stay in the hospital? (in days)
    """

    
    try:
        chat_completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print("LLM Response: ", chat_completion.choices[0].message['content'].strip())
    except Exception as e:
        print(f"An error occurred: {e}")

    # Collect inputs from the user
    user_input = {
        "age": int(input("Your age: ")),
        "gender": input("Your gender: "),
        "blood_type": input("Your blood type: "),
        "medical_condition": input("What medical condition are you experiencing?: "),
        "stay_period": int(input("How long do you expect to stay in the hospital (in days)?: "))
    }

    predicted_cost = predict_healthcare_cost(user_input)
    print(f"The estimated healthcare cost for your treatment is: ${predicted_cost:.2f}")

# Run the interactive session
chat_with_llm()
