import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score, ShuffleSplit
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import (KFold, cross_val_score, train_test_split,
                                     LeaveOneOut)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             classification_report, roc_auc_score, roc_curve, 
                             log_loss, accuracy_score)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
import io

st.title("CLASSIFICATION METRICS")

# Function to load the dataset of Heart Disease Data
@st.cache_data
def heart_disease(uploaded_file):
    names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
             'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
    dataframe = pd.read_csv(uploaded_file, names=names, header=0)
    
    # Encode the target variable
    le = LabelEncoder()
    dataframe['Heart Disease'] = le.fit_transform(dataframe['Heart Disease'])  # Convert 'Absence'/'Presence' to 0/1
    
    return dataframe

# Function to load the dataset of Forest Fire Data
@st.cache_data
def forest_fire(uploaded_file):
    names = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 
             'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
    dataframe = pd.read_csv(uploaded_file, names=names, header=0)
    
    # Convert month and day to numeric values for model processing
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    day_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
    
    dataframe['month'] = dataframe['month'].map(month_mapping)
    dataframe['day'] = dataframe['day'].map(day_mapping)
    
    return dataframe

# Function for user input in Heart Disease model
def get_heart_disease_input():
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    chest_pain_type = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    bp = st.number_input("Blood Pressure (BP)", min_value=0, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=0, value=200)
    fbs_over_120 = st.selectbox("FBS over 120 (0: No, 1: Yes)", [0, 1])
    ekg_results = st.selectbox("EKG Results (0-2)", [0, 1, 2])
    max_hr = st.number_input("Maximum Heart Rate", min_value=0, value=150)
    exercise_angina = st.selectbox("Exercise Angina (0: No, 1: Yes)", [0, 1])
    st_depression = st.number_input("ST Depression", min_value=0.0, value=0.0)
    slope_of_st = st.selectbox("Slope of ST (0-2)", [0, 1, 2])
    num_vessels_fluro = st.selectbox("Number of Vessels Fluro (0-3)", [0, 1, 2, 3])
    
    # Thallium Feature Selection
    thallium_choice = st.selectbox("Thallium (None, Normal, Abnormal)", 
                                   ["None (1-3)", "Normal (2-3)", "Abnormal (5-7)"])

    # Map the selection to the corresponding numerical range
    if thallium_choice == "None (1-3)":
        thallium = 1  # or any value from 1 to 3 depending on your dataset
    elif thallium_choice == "Normal (2-3)":
        thallium = 2  # or any value from 2 to 3
    elif thallium_choice == "Abnormal (5-7)":
        thallium = 5  # or any value from 5 to 7
    
    # Collect all features into an array
    features = np.array([[age, sex, chest_pain_type, bp, cholesterol, fbs_over_120, 
                          ekg_results, max_hr, exercise_angina, st_depression, 
                          slope_of_st, num_vessels_fluro, thallium]])
    
    return features

# Function for user input in Forest Fire model
def get_forest_fire_input():
    X = st.number_input("X-axis spatial coordinate within the Montesinho park map", min_value=0, max_value=9, value=5)
    Y = st.number_input("Y-axis spatial coordinate within the Montesinho park map", min_value=0, max_value=9, value=5)
    month = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day = st.selectbox("Day of the week", ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])
    FFMC = st.number_input("FFMC Index (Fine Fuel Moisture Code)", min_value=0.0, max_value=100.0, value=85.0)
    DMC = st.number_input("DMC Index (Duff Moisture Code)", min_value=0.0, max_value=200.0, value=25.0)
    DC = st.number_input("DC Index (Drought Code)", min_value=0.0, max_value=800.0, value=200.0)
    ISI = st.number_input("ISI Index (Initial Spread Index)", min_value=0.0, max_value=50.0, value=10.0)
    temp = st.number_input("Temperature (in Celsius)", min_value=-5.0, max_value=50.0, value=15.0)
    RH = st.number_input("Relative Humidity (%)", min_value=0, max_value=100, value=50)
    wind = st.number_input("Wind Speed (in km/h)", min_value=0.0, max_value=50.0, value=5.0)
    rain = st.number_input("Rain (in mm)", min_value=0.0, max_value=10.0, value=0.0)
    

    # Map month and day to numerical values
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    day_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
    
    # Convert month and day to their numeric values
    month_num = month_mapping[month]
    day_num = day_mapping[day]
    
    # Return the input features
    features = np.array([[X, Y, month_num, day_num, FFMC, DMC, DC, ISI, temp, RH, wind, rain]])
    return features



def main():
    st.title("Model Evaluation")

    model_choice = st.selectbox("Choose Area", ("Health", "Environment"))

    if model_choice == "Health":
        # Upload file
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            # Load the dataset
            st.write("Loading the dataset...")
            dataframe = heart_disease(uploaded_file)

            # Display the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            evaluation_choice = st.selectbox("Select Evaluation Method", 
                                             ["K-fold Cross Validation", 
                                              "Leave-One-Out Cross Validation", 
                                              "Prediction"])

            if evaluation_choice == "K-fold Cross Validation":
                st.subheader("K-fold Cross Validation")
                array = dataframe.values
                X = array[:, :-1]  # Features
                Y = array[:, -1]   # Target variable
                num_folds = st.slider("Select number of folds for KFold Cross Validation:", 2, 10, 5)
                kfold = KFold(n_splits=num_folds)
                model = LogisticRegression(max_iter=210)
                results = cross_val_score(model, X, Y, cv=kfold)
                st.write(f"Accuracy: {results.mean() * 100:.3f}%")
                st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

                # Metrics Calculation
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
                model.fit(X_train, Y_train)
                Y_prob = model.predict_proba(X_test)[:, 1]
                
                # Classification Metrics
                st.subheader("Classification Metrics")

                
                st.write(f"Logarithmic Loss: {log_loss(Y_test, Y_prob):.3f}")

                predicted = model.predict(X_test)

                # Classification Accuracy
                accuracy = accuracy_score(Y_test, predicted) 
                st.write(f"Classification Accuracy: {accuracy * 100:.3f}%")
                st.write("Confusion Matrix:")
                predicted = model.predict(X_test)
                matrix = confusion_matrix(Y_test, predicted)
                st.write(matrix)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(confusion_matrix=matrix).plot(cmap=plt.cm.Blues, ax=ax)
                st.pyplot(fig)

                # Display classification report
                st.write("Classification Report:")
                report = classification_report(Y_test, predicted, output_dict=True)
                report_df = pd.DataFrame(report).transpose()  # Convert to DataFrame
                st.dataframe(report_df) 

                st.write(f"ROC AUC Score: {roc_auc_score(Y_test, Y_prob):.3f}")

                # Plot ROC Curve
                fpr, tpr, _ = roc_curve(Y_test, Y_prob)
                plt.figure()
                plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc_score(Y_test, Y_prob))
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
                st.pyplot()

                

                # Save the trained model with unique key
                model_filename = "heart_disease.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    st.download_button("Download Trained Model", f, file_name=model_filename, key="download_kfold")

            elif evaluation_choice == "Leave-One-Out Cross Validation":
                st.subheader("Leave-One-Out Cross Validation (LOOCV)")
                array = dataframe.values
                X = array[:, :-1]  # Features
                Y = array[:, -1]   # Target variable
                loocv = LeaveOneOut()
                model = LogisticRegression(max_iter=500)
                results = cross_val_score(model, X, Y, cv=loocv)
                st.write(f"Accuracy: {results.mean() * 100:.3f}%")
                st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

                # Metrics Calculation
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
                model.fit(X_train, Y_train)
                Y_prob = model.predict_proba(X_test)[:, 1]
                
                # Classification Metrics
                st.subheader("Classification Metrics")
                st.write(f"Logarithmic Loss: {log_loss(Y_test, Y_prob):.3f}")

                
                predicted = model.predict(X_test)

                # Classification Accuracy
                accuracy = accuracy_score(Y_test, predicted) 
                st.write(f"Classification Accuracy: {accuracy * 100:.3f}%")
                st.write("Confusion Matrix:")
                predicted = model.predict(X_test)
                matrix = confusion_matrix(Y_test, predicted)
                st.write(matrix)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(confusion_matrix=matrix).plot(cmap=plt.cm.Blues, ax=ax)
                st.pyplot(fig)

                # Display classification report
                st.write("Classification Report:")
                report = classification_report(Y_test, predicted, output_dict=True)
                report_df = pd.DataFrame(report).transpose()  
                st.dataframe(report_df) 

                st.write(f"ROC AUC Score: {roc_auc_score(Y_test, Y_prob):.3f}")

                # Plot ROC Curve
                fpr, tpr, _ = roc_curve(Y_test, Y_prob)
                plt.figure()
                plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc_score(Y_test, Y_prob))
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
                st.pyplot()

                

                # Save the trained model with unique key
                model_filename = "heart_disease.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    st.download_button("Download Trained Model", f, file_name=model_filename, key="download_loocv")

            elif evaluation_choice == "Prediction":
                st.subheader("Prediction")
                st.write("Upload your trained model for prediction:")
                uploaded_model = st.file_uploader("Upload your trained model file", type=["pkl"])
                user_data = get_heart_disease_input() 
                if uploaded_model is not None:
                    loaded_model = joblib.load(uploaded_model)
                    prediction = loaded_model.predict(user_data) 
                    st.write("Prediction Result:")
                    st.write("Heart Disease" if prediction[0] == 1 else "No Heart Disease")

    if model_choice == "Environment":
        uploaded_file = st.file_uploader("Upload your Forest Fire CSV file", type=["csv"])

        if uploaded_file is not None:
            st.write("Loading the dataset...")
            dataframe = forest_fire(uploaded_file)

            if dataframe is not None:
                st.subheader("Dataset Preview")
                st.write(dataframe.head())

                selection = st.selectbox("Choose an evaluation method:", 
                                        ["Train-Test Split", "Repeated Random Test-Train Splits", "Prediction"])

                if selection == "Train-Test Split":
                    st.subheader("Split into Train and Test Sets")
                    test_size = st.slider("Test size (as a percentage)", 10, 50, 20) / 100
                    X = dataframe.drop('area', axis=1).values  # Features
                    Y = dataframe['area'].values  # Target variable (burned area)

                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
                    model = RandomForestRegressor()
                    model.fit(X, Y)

                    result = model.score(X_test, Y_test)
                    st.write(f"Accuracy: {result * 100:.3f}%")

                    # Regression Metrics
                    st.subheader("Regression Metrics")
                    Y_pred = model.predict(X_test)
                    mse = mean_squared_error(Y_test, Y_pred)
                    st.write(f"Mean Squared Error (MSE): {mse:.3f}")

                    mae = np.mean(np.abs(Y_test - Y_pred))
                    st.write(f"Mean Absolute Error (MAE): {mae:.3f}")

                    r2 = 1 - (np.sum((Y_test - Y_pred) ** 2) / np.sum((Y_test - np.mean(Y_test)) ** 2))
                    st.write(f"R-squared (R²): {r2:.3f}")

                    # Save the trained model
                    model_filename = "forest_fire_splitmodel.pkl"
                    joblib.dump(model, model_filename)
                    with open(model_filename, "rb") as f:
                        st.download_button("Download Trained Model", f, file_name=model_filename)

                elif selection == "Repeated Random Test-Train Splits":
                    st.subheader("Repeated Random Test-Train Splits")

                    X = dataframe.drop(columns='area').values
                    Y = dataframe['area'].values

                    n_splits = st.slider("Select number of splits:", 2, 20, 10)
                    test_size = st.slider("Select test size proportion:", 0.1, 0.5, 0.33)
                    shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size)

                    model = RandomForestRegressor()
                    mse_scores = []
                    model.fit(X, Y)
                    results = cross_val_score(model, X, Y, cv=shuffle_split)
                    result = model.score(X, Y)
                    st.write(f"Accuracy: {result * 100:.3f}%")
                    st.write(f"Standard Deviation: {results.std():.3f}")
                    
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
                    # Fit the model on the complete dataset for regression metrics
                    model.fit(X, Y)
                    Y_pred = model.predict(X_test)

                    # Regression Metrics
                    st.subheader("Regression Metrics")
                    mse = mean_squared_error(Y_test, Y_pred)
                    mse_scores.append(mse)
                    st.write(f"Mean Squared Error (MSE): {mse:.3f}")

                    mae = np.mean(np.abs(Y_test- Y_pred))
                    st.write(f"Mean Absolute Error (MAE): {mae:.3f}")

                    r2 = 1 - (np.sum((Y_test - Y_pred) ** 2) / np.sum((Y_test - np.mean(Y_test)) ** 2))
                    st.write(f"R-squared (R²): {r2:.3f}")

                    model_filename = "forest_fire.pkl"
                    joblib.dump(model, model_filename)
                    with open(model_filename, "rb") as f:
                        st.download_button("Download Trained Model", f, file_name=model_filename)

                elif selection == "Prediction":
                    st.subheader("Prediction")
                    st.write("Upload your trained model for prediction:")
                    uploaded_model = st.file_uploader("Upload your trained model file", type=["pkl"])
                    user_data = get_forest_fire_input()

                    if uploaded_model is not None:
                        loaded_model = joblib.load(uploaded_model)
                        prediction = loaded_model.predict(user_data)
                        st.write("Prediction Result:")
                        st.write(f"Predicted Burned Area: {prediction[0]:.3f} ha")

if __name__ == "__main__":
    main()