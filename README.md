**Mental Health Treatment Predictor**

A Machine Learning web application that predicts whether an individual is likely to seek mental health treatment based on workplace and personal factors. The project integrates model development, explainability, and automated CI/CD deployment using Azure DevOps and Flask.


**Project Overview**

-This application is designed to:

-Predict if a person is likely to seek mental health treatment based on survey responses.

-Visualize key factors affecting mental health decisions.

-Provide a simple web interface for real-time predictions.

-Automate deployment using Azure DevOps (CI/CD pipeline).



**Key Features**


-Built using six machine learning models:
  • Logistic Regression
  • Random Forest
  • XGBoost
  • Gradient Boosting (Final model)
  • Support Vector Machine (SVM)
  • Linear Discriminant Analysis (LDA)

-Visual interpretability using SHAP values

-Flask-based user interface for predictions

-Azure-hosted using CI/CD via DevOps pipelines

-API tested with Postman for validation




**Project Workflow**
 
-This project was implemented through a combination of manual machine learning development and automated cloud deployment using Azure DevOps. Below is the step-by-step workflow:

**Manual Development Process**

   
-Dataset Search & Selection:
 Identified and selected a publicly available mental health in tech workplace dataset. Assessed feature relevance (e.g., age, benefits, family history, work interference).

-Data Preprocessing:
Handled missing values using logical imputation. Applied label encoding to categorical variables.

-Exploratory Data Analysis (EDA) :
Visualized trends and patterns using heatmaps, KDE plots, and grouped bar charts. Discovered correlations between mental health treatment and features like work interference and benefits.

-Model Development :
Trained six classification models: Logistic Regression, Random Forest, XGBoost (initial), followed by Gradient Boosting, SVM, and LDA. Evaluated them using 5-fold Stratified Cross-Validation.

-Model Tuning :
Used GridSearchCV to optimize hyperparameters for XGBoost and Gradient Boosting.

-Model Explainability (SHAP Values) :
Applied SHAP to interpret feature contributions. Generated SHAP summary and force plots to visualize how individual features affect prediction output.

-Model Serialization :
Saved the trained model (model.pkl) and preprocessing pipeline (transform.pkl) using pickle for deployment.

-Flask Web App Development :
Developed a Flask-based web application for input and prediction. Created an HTML form to take user inputs and return predictions with a confidence score.

-Docker Containerization :
Created a Dockerfile and requirements.txt for containerizing the Flask app. Successfully built and tested the container locally.



**Automated CI/CD Using Azure DevOps**

-Source Code Management:
Pushed project code to Azure Repos for version control and integration.

-Pipeline Setup:
Configured an Azure DevOps pipeline using azure-pipelines.yml with the following stages:

-Build: Set up the environment and install dependencies

-Test: Run basic validations

-Deploy: Publish the app to Azure App Service

-App:
Deployed both demo and enhanced model versions via pipeline to Azure App Service.

-Postman API Testing:
Used Postman to test the deployed API endpoint by sending input JSON payloads. Verified predictions, status codes, and response structure.

**Live Demo
Link to deployed app:**


https://employeewise-gtc5c3eqajcghdh0.eastasia-01.azurewebsites.net/



How to Use (User Manual)

1. Open the App
 Visit the deployed app using the link above.

2. Fill the Form
   Enter the following information:

    -Age
    -Family History (Yes/No)
    -Remote Work (Yes/No)
    -Benefits Provided (Yes/No/Don't know)
    -Care Options (Yes/No/Not sure)
    -Work Interference (Often/Sometimes/Never/etc.)
    -Supervisor Support (Yes/No/Somewhat)

   
3. Submit
   Click "Predict" to submit your inputs.

4. View Result
   
   The app will display:

      -Prediction: Likely to seek treatment / Not likely

      -Confidence Score: e.g., 77%






