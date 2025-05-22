import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import os

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
    else:
        return None

# Function to create target variable
def create_target(df):
    df['OverallScore'] = (df['MathScore'] + df['ReadingScore'] + df['WritingScore']) / 3
    return df

# Function to train model
def train_model(X, y, test_size=0.2, n_estimators=100):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Create and train the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=n_estimators, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return model, X_train, X_test, y_train, y_test, y_pred, mse, rmse, r2, categorical_features, numerical_features

# Function for Data Overview page
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
        
        # Check for special columns
        if 'Unnamed: 0' in data.columns:
            if st.button("Remove 'Unnamed: 0' column"):
                data.drop('Unnamed: 0', axis=1, inplace=True)
                st.success("Column removed!")
                st.dataframe(data.head())
                st.session_state.data = data
        
        # Statistical Summary
        st.subheader("Statistical Summary")
        st.write(data.describe())
        
        # Create target variable if scores exist
        required_scores = ['MathScore', 'ReadingScore', 'WritingScore']
        if all(score in data.columns for score in required_scores):
            if 'OverallScore' not in data.columns:
                data = create_target(data)
                st.session_state.data = data
                st.success("Created 'OverallScore' as target variable")
            
            # Prepare data for model
            st.session_state.X = data.drop(['MathScore', 'ReadingScore', 'WritingScore', 'OverallScore'], axis=1)
            st.session_state.y = data['OverallScore']
            st.success("Data is ready for analysis and modeling!")
        else:
            st.warning("Could not find required score columns (MathScore, ReadingScore, WritingScore)")

# Function for Data Analysis page
def data_analysis():
    st.header("Data Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload data in the Data Overview page first.")
        return

    data = st.session_state.data

    # Various Analysis Options
    analysis_option = st.selectbox(
        "Choose Analysis Type",
        ["Score Distribution", "Gender Analysis", "Ethnic Group Analysis", "Parent Education Impact", 
         "Test Prep Impact", "Correlation Analysis"]
    )
    
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
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                gender_counts = data['Gender'].value_counts()
                ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90,
                       colors=['lightblue', 'lightpink'])
                ax.set_title('Gender Distribution')
                st.pyplot(fig)
            
            with col2:
                score_by_gender = data.groupby('Gender')[['MathScore', 'ReadingScore', 'WritingScore', 'OverallScore']].mean()
                st.write("Average Scores by Gender:")
                st.write(score_by_gender)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                score_by_gender.plot(kind='bar', ax=ax)
                ax.set_title('Average Scores by Gender')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        else:
            st.warning("Gender column not found in the dataset.")
    
    elif analysis_option == "Ethnic Group Analysis":
        st.subheader("Ethnic Group Analysis")
        
        if 'EthnicGroup' in data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                ethnic_counts = data['EthnicGroup'].value_counts()
                ax.bar(ethnic_counts.index, ethnic_counts.values, color='skyblue')
                ax.set_title('Ethnic Group Distribution')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                score_by_ethnic = data.groupby('EthnicGroup')['OverallScore'].mean().sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                score_by_ethnic.plot(kind='bar', ax=ax, color='lightgreen')
                ax.set_title('Average Overall Score by Ethnic Group')
                ax.set_ylabel('Overall Score')
                plt.xticks(rotation=45)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        else:
            st.warning("EthnicGroup column not found in the dataset.")
    
    elif analysis_option == "Parent Education Impact":
        st.subheader("Parent Education Impact")
        
        if 'ParentEduc' in data.columns:
            # Sort by education level if possible
            education_order = ["some high school", "high school", "some college", "associate's degree", 
                            "bachelor's degree", "master's degree"]
            
            educ_data = data.copy()
            if all(level in educ_data['ParentEduc'].unique() for level in education_order):
                educ_data['ParentEduc'] = pd.Categorical(
                    educ_data['ParentEduc'], 
                    categories=education_order, 
                    ordered=True
                )
            
            score_by_educ = educ_data.groupby('ParentEduc')[['MathScore', 'ReadingScore', 'WritingScore', 'OverallScore']].mean()
            
            st.write("Average Scores by Parent Education Level:")
            st.write(score_by_educ)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            score_by_educ.plot(kind='bar', ax=ax)
            ax.set_title('Average Scores by Parent Education Level')
            ax.set_ylabel('Score')
            plt.xticks(rotation=45)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("ParentEduc column not found in the dataset.")
    
    elif analysis_option == "Test Prep Impact":
        st.subheader("Test Preparation Impact")
        
        if 'TestPrep' in data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                prep_counts = data['TestPrep'].value_counts()
                ax.pie(prep_counts, labels=prep_counts.index, autopct='%1.1f%%', startangle=90,
                       colors=['lightgreen', 'lightcoral'])
                ax.set_title('Test Preparation Distribution')
                st.pyplot(fig)
            
            with col2:
                score_by_prep = data.groupby('TestPrep')[['MathScore', 'ReadingScore', 'WritingScore', 'OverallScore']].mean()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                score_by_prep.plot(kind='bar', ax=ax)
                ax.set_title('Average Scores by Test Preparation')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Calculate improvement percentage
                if len(score_by_prep) == 2 and 'completed' in score_by_prep.index and 'none' in score_by_prep.index:
                    improvement = ((score_by_prep.loc['completed'] - score_by_prep.loc['none']) / score_by_prep.loc['none'] * 100)
                    st.subheader("Test Prep Improvement:")
                    for subject, value in improvement.items():
                        st.write(f"{subject}: {value:.1f}% improvement with test prep")
        else:
            st.warning("TestPrep column not found in the dataset.")
    
    elif analysis_option == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        
        # Select only numeric columns for correlation
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = numeric_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Between Variables")
        st.pyplot(fig)
        
        st.write("Correlation Matrix:")
        st.write(corr_matrix)

# Function for Model Training page
def model_training():
    st.header("Model Training")
    
    if st.session_state.X is None or st.session_state.y is None:
        st.warning("Please upload and prepare data in the Data Overview page first.")
        return
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
        n_estimators = st.slider("Number of Trees in Random Forest", 50, 500, 100, 10)
    
    with col2:
        st.write("Features for Prediction:")
        st.write(f"Number of features: {st.session_state.X.shape[1]}")
        with st.expander("View features"):
            st.write(st.session_state.X.columns.tolist())
    
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            model, X_train, X_test, y_train, y_test, y_pred, mse, rmse, r2, cat_features, num_features = train_model(
                st.session_state.X, st.session_state.y, test_size, n_estimators
            )
            
            st.session_state.model = model
            st.session_state.categorical_features = cat_features
            st.session_state.numerical_features = num_features
            
            st.success("Model trained successfully!")
            
            # Display metrics
            st.subheader("Model Performance")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error", f"{mse:.4f}")
            col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
            col3.metric("R² Score", f"{r2:.4f}")
            
            # Plot actual vs predicted
            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Actual Overall Score')
            ax.set_ylabel('Predicted Overall Score')
            ax.set_title('Prediction vs Actual')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Save model option
            if st.button("Save Model"):
                try:
                    if not os.path.exists("models"):
                        os.makedirs("models")
                    joblib.dump(model, "models/student_performance_model.pkl")
                    st.success("Model saved successfully to 'models/student_performance_model.pkl'")
                except Exception as e:
                    st.error(f"Error saving model: {e}")
    
    else:
        if st.session_state.model is not None:
            st.info("Model already trained. You can train again with different parameters or proceed to Prediction.")

# Function for Prediction page
def prediction():
    st.header("Student Performance Prediction")
    
    if st.session_state.model is None:
        st.warning("Please train a model in the Model Training page first.")
        return
    
    st.subheader("Enter Student Information")
    
    # Load model from file option
    use_saved_model = st.checkbox("Use saved model instead", False)
    if use_saved_model:
        try:
            model_path = st.text_input("Model path", "models/student_performance_model.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                st.success("Model loaded successfully!")
            else:
                st.error(f"Model file not found: {model_path}")
                st.info("Using currently trained model instead.")
                model = st.session_state.model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Using currently trained model instead.")
            model = st.session_state.model
    else:
        model = st.session_state.model
    
    # Get original data to extract categories
    data = st.session_state.data
    
    # Create form for user input
    with st.form("prediction_form"):
        # For each categorical feature
        cat_inputs = {}
        for feature in st.session_state.categorical_features:
            unique_values = data[feature].unique().tolist()
            cat_inputs[feature] = st.selectbox(f"Select {feature}", unique_values)
        
        # For each numerical feature (if any)
        num_inputs = {}
        for feature in st.session_state.numerical_features:
            min_val = float(data[feature].min())
            max_val = float(data[feature].max())
            default_val = float(data[feature].mean())
            num_inputs[feature] = st.slider(f"Select {feature}", min_val, max_val, default_val)
        
        # Submit button
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Create input DataFrame
            input_data = {}
            input_data.update(cat_inputs)
            input_data.update(num_inputs)
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                
                # Display prediction
                st.success(f"Predicted Overall Score: {prediction:.2f}")
                
                # Calculate individual subject predictions (approximation)
                avg_ratios = data[['MathScore', 'ReadingScore', 'WritingScore']].mean() / data['OverallScore'].mean()
                
                math_pred = prediction * avg_ratios['MathScore']
                reading_pred = prediction * avg_ratios['ReadingScore']
                writing_pred = prediction * avg_ratios['WritingScore']
                
                # Display individual predictions
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Math Score", f"{math_pred:.2f}")
                col2.metric("Predicted Reading Score", f"{reading_pred:.2f}")
                col3.metric("Predicted Writing Score", f"{writing_pred:.2f}")
                
                # Performance category
                if prediction >= 80:
                    performance = "Excellent"
                    color = "green"
                elif prediction >= 70:
                    performance = "Good"
                    color = "blue"
                elif prediction >= 60:
                    performance = "Average"
                    color = "orange"
                else:
                    performance = "Needs Improvement"
                    color = "red"
                
                st.markdown(f"<h3 style='color:{color}'>Performance Category: {performance}</h3>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

# Main function to control the app flow
def main():
    if page == "Data Overview":
        data_overview()
    elif page == "Data Analysis":
        data_analysis()
    elif page == "Model Training":
        model_training()
    elif page == "Prediction":
        prediction()

# Run the app
if __name__ == "__main__":
    main()

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
This app predicts student performance based on various demographic and social factors.
""")