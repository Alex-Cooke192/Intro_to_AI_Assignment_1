import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

# Define global variables
df = pd.DataFrame()
le = LabelEncoder()
model = None
data_encoded_flag = False

# Parses csv or excel file and saves as a pandas DataFrame, clean data for training
def load_dataset():
    global df, data_encoded_flag
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
        ### Clean and format the data
            # Remove rows with missing data
            df = df.dropna()

            # Remove rows where age is unknown
            df = df[df['age'] != 'unknown']

            # Remove rows where study time varies
            df = df[df['study_hours_per_day'] != 'varies']

            # Remove rows where exam score > 100 - not possible! 
            df = df[df['exam_score'] < 101]

            # Remove leading/trailing spaces from text-based attributes
            data_encoded_flag = False
            for column in df.columns:
                if df[column].dtype == object:
                    df[column] = df[column].str.strip()
    
        ### Get data to correct types
            # convert age to integers
            df["age"] = pd.to_numeric(df["age"], errors="coerce")

            # Remove any rows where 'age' is NaN
            df = df.dropna(subset=["age"])
                
            # Convert study_hours_per_day to number
            df["study_hours_per_day"] = pd.to_numeric(df["study_hours_per_day"], errors="coerce")
            
            # Remove any rows where 'study_hours_per_day' is NaN
            df = df.dropna(subset=["study_hours_per_day"])
            
            # Output to GUI
            messagebox.showinfo("Success", "Dataset loaded successfully *but did you check the script!")
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    return None

def encode_data(features, target):
    global le, data_encoded_flag, model
    
    # Create a copy of feature variable data
    X = df[features].copy()

    # Encode only categorical columns
    for column in X.columns:
        if X[column].dtype == 'object':  # Only transform non-numeric features
            X[column] = le.fit_transform(X[column])

    # Handle the target variable
    print('target type:', df[target].dtype)
    if df[target].dtype == 'object':  # If categorical, encode 
        y = le.fit_transform(df[target])
    else:
        print('Data continuous')
        y = df[target].values  # If continuous, keep as-is
    
    data_encoded_flag = True
    return X, y

# Uses input data to create a model and tests accuracy of the model
def train_model(df, features, target):
    global model
    if df[target].dtype == 'object':  # categorical
        model = RandomForestClassifier()
        train_model_classifier(features, target)
    else: # Continuous data
        model = RandomForestRegressor()
        train_model_regression(features,target)

# Used to train model if target data is categoric
def train_model_classifier(features, target):
    print('Classifier Training...')
    global df, le, model, data_encoded_flag
    try:
        # Label encode categorical data
        X,y = encode_data(features, target)

        # Ensure target data is encoded
        if df[target].dtype == 'object':
            y = le.fit_transform(df[target])

        # Split data into training/testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestClassifier()
        model = model.fit(X_train, y_train)

        # Calculate accurcy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Model Trained", f"Model trained successfully! Accuracy: {accuracy:.2f}")
        
        # Return trained model
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")
    return None

# Used to train model if target data is continouous (more likely) 
def train_model_regression(features, target):
    global df, le, model, data_encoded_flag
    try:
        # Label encode categorical data
        X,y = encode_data(features, target)

        # Split into training/testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit data to model and predict values 
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate accuracy score (r2 score as data is continuous)
        accuracy = r2_score(y_test, y_pred)

        # Return to user 
        messagebox.showinfo("Model Trained", f"Model trained successfully! Accuracy: {accuracy:.2f}")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")
    return None

# Determine if target data is continuous or categorical to determine prediction function
def make_predictions(features, target):
    if df[target].dtype == 'object':  # categorical
        print('Predicted using classifier')
        make_predictions_classifier(features, target)
    else: # Continuous data
        print('Predicted using regressor')
        make_predictions_regression(features,target)


def make_predictions_regression(features, target):
    global model, df, le, data_encoded_flag 
    # Encode raw data
    X_new, _ = encode_data(features, target)

    # Make predictions
    predictions = model.predict(X_new)

    # Display results in GUI 
    student_IDs = df["student_id"]
    for element in predictions:
        print(element)
        result_text.insert(tk.END, f"{student_IDs.iloc[element]}:{predictions[element]}\n")
    return True

def make_predictions_classifier(features, target):
    global model, df, le, data_encoded_flag
    try:
        # Check if model has been trained
        if hasattr(model, "estimators_") == False:
            messagebox.showerror("Error", "Model is not trained. Please train the model first.")
            return False
        
        # Encode raw data - ONLY CATEGORICAL DATA
        X_new, _ = encode_data(features, target)
        
        # Make predictions
        predictions = model.predict(X_new) 

        # Decode predictions
        results = le.inverse_transform(predictions)
        
        # Display results in GUI 
        student_IDs = df["student_id"]
        for value in predictions:
            result_text.insert(tk.END, f"{student_IDs.iloc[value]}:{predictions[value]}\n")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {e}")

# Set up main tkinter window 
root = tk.Tk()
root.title("Student Predictive Grades")

# Create load_dataset button and add to GUI
load_button = tk.Button(root, text="Load Dataset", command=lambda: load_dataset())
load_button.pack(pady=10)

# Add label & entry widget for feature variable(s)
tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root)
features_entry.pack(pady=5)

# Add label & entry widget for target variables
tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root)
target_entry.pack(pady=5)

# Create train_model button & add to GUI
train_button = tk.Button(root, text="Train Model", command=lambda: train_model(df, features_entry.get().split(','), target_entry.get()))
train_button.pack(pady=10)

# Create make_predictions button & add to GUI
predict_button = tk.Button(root, text="Make Predictions", command=lambda: make_predictions(model, df, features_entry.get().split(',')))
predict_button.pack(pady=10)

# Add box for predictions
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# Run tkinter window
root.mainloop()

