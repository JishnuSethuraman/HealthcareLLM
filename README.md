# Healthcare Cost Prediction with LLM and TabNet

This project integrates a **TabNet** model for healthcare cost prediction with a **Large Language Model (LLM)** interface using OpenAI's GPT. The LLM collects user inputs (like age, gender, symptoms, etc.) and provides a cost prediction for healthcare services based on those inputs.

## Project Structure

- **`LLMimplementation.py`**: This script interacts with the OpenAI API to collect user inputs via a chatbot interface and then uses the TabNet model to predict healthcare costs.
- **`trainModel.py`**: train the model.
- **`encoders.pkl`** and **`scaler.pkl`**: The encoders and scaler used for preprocessing the input data before feeding it to the TabNet model.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.7 or above
- Necessary Python libraries: 
  ```bash
  pip install -r requirements.txt
