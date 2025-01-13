import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = "AIzaSyDsPRBlDBTwgWJERKDdj87iOps8T_9lx8w"
genai.configure(api_key=api_key)

DEFAULT_FILE_PATH = "sample_financial_market_data.csv"

dataset_summary = None

@st.cache_data
def load_and_preprocess_data(file_path):
    global dataset_summary
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            st.error("The loaded dataset is empty. Please provide a valid dataset.")
            return None, None

        st.write("Initial dataset shape:", df.shape)
        for col in df.columns:
            if col != "target":
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if df[col].isnull().any():
                        st.warning(f"Column '{col}' contains non-numeric values. Filling missing values with mean.")
                        df[col] = df[col].fillna(df[col].mean())
                except Exception as e:
                    st.warning(f"Skipping column '{col}' due to conversion error: {e}")
                    df = df.drop(columns=[col])

        if df.empty:
            st.error("All data has been removed after preprocessing. Ensure the dataset contains valid numeric values.")
            return None, None

        if "target" not in df.columns:
            st.warning("'Target' column missing. Generating a dummy target column for demonstration purposes.")
            df["target"] = (df.iloc[:, 0] > df.iloc[:, 0].median()).astype(int)

        st.write("Processed dataset shape:", df.shape)
        dataset_summary = (
            f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
            f"The target distribution is as follows: {df['target'].value_counts().to_dict()}. "
            f"The key features include: {', '.join(df.drop(columns=['target']).columns.tolist())}."
        )
        features = df.drop(columns=["target"])
        target = df["target"]
        return features, target

    except Exception as e:
        st.error(f"Error processing dataset: {e}")
        return None, None

@st.cache_resource
def train_model(features, target):
    try:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.success("Model trained successfully!")
        st.json(report)
        return model
    except Exception as e:
        st.error(f"Error training the model: {e}")
        return None

def query_gemini(prompt):
    try:
        context = (
            f"You are an AI assistant specializing in financial investments. Use the following dataset context for all responses:\n\n"
            f"{dataset_summary}\n\n"
            f"Respond concisely and stay on topic about investment strategies, data analysis, and market trends."
            f"Do not be generic, provide specific and detailed responses according to the market situation."
        )
        full_prompt = f"{context}\n\n{prompt}"
        model = genai.GenerativeModel(model_name='models/gemini-1.5-pro')
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error querying Gemini API: {e}"

def main():
    st.title("Anomaly Detection and Investment Strategy App")
    st.header("Upload a CSV File")
    uploaded_file = st.file_uploader("Upload your CSV file for analysis", type=["csv"])

    if uploaded_file:
        file_path = uploaded_file
    else:
        st.info("Using default dataset.")
        file_path = DEFAULT_FILE_PATH

    features, target = load_and_preprocess_data(file_path)

    model = None
    if features is not None and target is not None:
        st.header("Dataset Overview")
        st.write("### Features and Target Distribution")
        st.write(f"Dataset Shape: {features.shape}")
        st.write(f"Target Distribution: {target.value_counts().to_dict()}")

        st.header("Model Training")
        model = train_model(features, target)

    if model:
        st.header("Market Condition Prediction")
        st.write("Enter feature values to predict market anomalies.")
        user_inputs = {}
        for col in features.columns:
            user_inputs[col] = st.number_input(f"{col}", value=0.0, step=0.1)

        if st.button("Get Prediction and Recommendation"):
            try:
                input_data = pd.DataFrame([user_inputs])
                prediction = model.predict(input_data)[0]
                prediction_label = "Crash" if prediction == 1 else "No Crash"
                st.success(f"Model Prediction: {prediction_label}")
                prompt = (
                    f"Given the following financial data:\n"
                    f"{', '.join([f'{col}: {val}' for col, val in user_inputs.items()])}\n\n"
                    f"The model predicts: {prediction_label}.\n"
                    "Provide a detailed investment strategy based on this prediction."
                )
                gemini_response = query_gemini(prompt)
                st.info(f"Investment Recommendation:\n{gemini_response}")
            except Exception as e:
                st.error(f"Error during prediction or recommendation: {e}")

        st.header("Chat with AI Investment Agent")
        st.write("Ask questions about investments or the market.")
        user_question = st.text_input("Your Question")
        if st.button("Send"):
            try:
                response = query_gemini(user_question)
                st.success("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error querying Gemini: {e}")

if __name__ == "__main__":
    main()