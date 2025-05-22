# Student_Performance_Prediction

A Streamlit-based web application that predicts student performance based on demographic and social factors using machine learning.


## 📚 Project Overview

**Student Performance Prediction System** is an interactive web application developed using **Streamlit** that enables educators, researchers, and data enthusiasts to analyze and predict student academic performance. The application utilizes machine learning techniques (Random Forest Regression) to model the relationship between Educational institutions collect large amounts of data on students—such as gender, ethnicity, parental education level, test preparation, and academic scores. However, this data is often underutilized when it comes to identifying students at risk of underperforming or in need of additional support.




This project addresses the following key problems:

- Can we predict a student’s performance using non-academic and academic indicators?
- Which factors are most influential in determining performance?
- How can we visualize these trends to provide actionable insights to educators and stakeholders?



Goal: Provide an interpretable, accessible machine learning interface where users can:

- Upload real student datasets.
- Explore the data visually.
- Train a predictive model.
- Input new student attributes to get estimated scores and performance level.


This tool is especially useful for:

- Educators seeking to identify at-risk students.
- Policy-makers evaluating factors influencing student success.
- Data science learners exploring real-world regression use cases.
- Schools and institutions looking to make data-informed decisions.


## 🧠 Key Features

- **📁 Data Upload and Inspection**: Upload student data in CSV format and get immediate previews, summaries, and null value detection.
- **📊 Dynamic Data Visualization**: Analyze score trends and insights across gender, ethnic group, parental education, and more.
- **🎯 Target Engineering**: Automatically calculates an `OverallScore` from core subjects.
- **🔧 Model Training Interface**: Train a Random Forest model with adjustable test size and tree count.
- **🧮 Prediction Engine**: Predict performance for individual students based on custom inputs.
- **📈 Performance Feedback**: Shows predicted scores with performance category indicators.
- **💾 Save & Load Models**: Save trained models for future predictions.



## 🚀 Getting Started

### Installation

pip install -r requirements.txt


📦 Dependencies
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

💡 Value Proposition
This system empowers non-technical users (like teachers or school admins) to:
- Analyze which social/demographic factors are impacting student outcomes
- Predict future performance without needing coding or ML experience
- Make data-informed decisions to provide support where it matters most

