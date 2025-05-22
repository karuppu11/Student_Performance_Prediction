import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Import model functions from model.py
from model import train_model, create_target

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)

# App title and description
st.title("📊 Student Performance Prediction System")
st.markdown("""
This application predicts student performance based on various demographic and social factors.
Upload your student data, analyze it, and predict scores using machine learning.
""")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Data Overview", "Data Analysis", "Model Training", "Prediction"])

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'categorical_features' not in st.session_state:
    st.session_state.categorical_features = None
if 'numerical_features' not in st.session_state:
    st.session_state.numerical_features = None

# Function to load data
def load_data():
    uploaded_file = st.file_uploader("Upload your student data CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error: {e}")
    return None

# Data Overview Page
def data_overview():
    st.header("Data Overview")
    data = load_data()
    if data is not None:
        st.session_state.data = data
        st.subheader("Raw Data Preview")
        st.dataframe(data.head())

        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Total Records: {data.shape[0]}")
            st.write(f"Total Features: {data.shape[1]}")
        with col2:
            st.write("Missing Values:")
            st.write(data.isnull().sum())

        if 'Unnamed: 0' in data.columns:
            if st.button("Remove 'Unnamed: 0' column"):
                data.drop('Unnamed: 0', axis=1, inplace=True)
                st.success("Column removed!")
                st.dataframe(data.head())
                st.session_state.data = data

        st.subheader("Statistical Summary")
        st.write(data.describe())

        required_scores = ['MathScore', 'ReadingScore', 'WritingScore']
        if all(score in data.columns for score in required_scores):
            if 'OverallScore' not in data.columns:
                data = create_target(data)
                st.session_state.data = data
                st.success("Created 'OverallScore' as target variable")

            st.session_state.X = data.drop(['MathScore', 'ReadingScore', 'WritingScore', 'OverallScore'], axis=1)
            st.session_state.y = data['OverallScore']
            st.success("Data is ready for analysis and modeling!")
        else:
            st.warning("Could not find required score columns (MathScore, ReadingScore, WritingScore)")

# Data Analysis Page
def data_analysis():
    st.header("Data Analysis")
    if st.session_state.data is None:
        st.warning("Please upload data in the Data Overview page first.")
        return

    data = st.session_state.data
    analysis_option = st.selectbox("Choose Analysis Type", [
        "Score Distribution", "Gender Analysis", "Ethnic Group Analysis",
        "Parent Education Impact", "Test Prep Impact", "Correlation Analysis"
    ])

    if analysis_option == "Score Distribution":
        st.subheader("Score Distribution")
        score_type = st.selectbox("Select Score Type", ["MathScore", "ReadingScore", "WritingScore", "OverallScore"])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[score_type], kde=True, bins=20, color='skyblue', ax=ax)
        ax.set_title(f'Distribution of {score_type}')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Mean: {data[score_type].mean():.2f}")
            st.write(f"Median: {data[score_type].median():.2f}")
        with col2:
            st.write(f"Min: {data[score_type].min():.2f}")
            st.write(f"Max: {data[score_type].max():.2f}")

    elif analysis_option == "Gender Analysis":
        st.subheader("Gender Analysis")
        if 'Gender' in data.columns:
            fig, ax = plt.subplots(figsize=(6, 6))
            data['Gender'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['lightblue', 'lightpink'])
            ax.set_ylabel('')
            ax.set_title('Gender Distribution')
            st.pyplot(fig)

            st.write("Average Scores by Gender")
            st.write(data.groupby('Gender')[['MathScore', 'ReadingScore', 'WritingScore', 'OverallScore']].mean())

    elif analysis_option == "Ethnic Group Analysis":
        st.subheader("Ethnic Group Analysis")
        if 'EthnicGroup' in data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=data['EthnicGroup'].value_counts().index,
                        y=data['EthnicGroup'].value_counts().values, ax=ax)
            ax.set_title("Ethnic Group Distribution")
            ax.set_ylabel("Count")
            ax.set_xlabel("Ethnic Group")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.write("Average Scores by Ethnic Group")
            st.write(data.groupby('EthnicGroup')[['MathScore', 'ReadingScore', 'WritingScore', 'OverallScore']].mean())

    elif analysis_option == "Parent Education Impact":
        st.subheader("Parent Education Impact")
        if 'ParentEduc' in data.columns:
            edu_order = [
                "some high school", "high school", "some college",
                "associate's degree", "bachelor's degree", "master's degree"
            ]
            if all(e in data['ParentEduc'].unique() for e in edu_order):
                data['ParentEduc'] = pd.Categorical(data['ParentEduc'], categories=edu_order, ordered=True)

            avg_scores = data.groupby('ParentEduc')[['MathScore', 'ReadingScore', 'WritingScore', 'OverallScore']].mean()
            st.write(avg_scores)

            fig, ax = plt.subplots(figsize=(12, 6))
            avg_scores.plot(kind='bar', ax=ax)
            ax.set_title('Scores by Parent Education Level')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    elif analysis_option == "Test Prep Impact":
        st.subheader("Test Preparation Impact")
        if 'TestPrep' in data.columns:
            st.write("Average Scores by Test Preparation:")
            st.write(data.groupby('TestPrep')[['MathScore', 'ReadingScore', 'WritingScore', 'OverallScore']].mean())

            fig, ax = plt.subplots(figsize=(8, 6))
            data.groupby('TestPrep')['OverallScore'].mean().plot(kind='bar', ax=ax)
            ax.set_title('Impact of Test Preparation')
            ax.set_ylabel('Overall Score')
            st.pyplot(fig)

    elif analysis_option == "Correlation Analysis":
        st.subheader("Correlation Matrix")
        corr = data.select_dtypes(include=['int64', 'float64']).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# Model Training Page
def model_training():
    st.header("Model Training")
    if st.session_state.X is None or st.session_state.y is None:
        st.warning("Please upload and prepare data in the Data Overview page first.")
        return

    test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
    n_estimators = st.slider("Number of Trees", 50, 500, 100, step=10)

    if st.button("Train Model"):
        with st.spinner("Training..."):
            model, X_train, X_test, y_train, y_test, y_pred, mse, rmse, r2, cat_features, num_features = train_model(
                st.session_state.X, st.session_state.y, test_size, n_estimators
            )
            st.session_state.model = model
            st.session_state.categorical_features = cat_features
            st.session_state.numerical_features = num_features

            st.success("Model trained successfully!")
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("MSE", f"{mse:.4f}")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Prediction vs Actual")
            st.pyplot(fig)

            if st.button("Save Model"):
                if not os.path.exists("models"):
                    os.makedirs("models")
                joblib.dump(model, "models/student_performance_model.pkl")
                st.success("Model saved as 'models/student_performance_model.pkl'")

# Prediction Page
def prediction():
    st.header("Student Performance Prediction")
    if st.session_state.model is None:
        st.warning("Please train or load a model first.")
        return

    data = st.session_state.data
    model = st.session_state.model

    with st.form("prediction_form"):
        input_data = {}
        for col in st.session_state.categorical_features:
            input_data[col] = st.selectbox(f"{col}", data[col].dropna().unique())

        for col in st.session_state.numerical_features:
            min_val, max_val = float(data[col].min()), float(data[col].max())
            default = float(data[col].mean())
            input_data[col] = st.slider(f"{col}", min_val, max_val, default)

        submitted = st.form_submit_button("Predict")
        if submitted:
            df_input = pd.DataFrame([input_data])
            try:
                pred = model.predict(df_input)[0]
                st.success(f"Predicted Overall Score: {pred:.2f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# Main
def main():
    if page == "Data Overview":
        data_overview()
    elif page == "Data Analysis":
        data_analysis()
    elif page == "Model Training":
        model_training()
    elif page == "Prediction":
        prediction()

if __name__ == "__main__":
    main()

st.sidebar.markdown("---")
st.sidebar.info("This app predicts student performance using machine learning.")
