import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Define global variables
df = pd.DataFrame()
le = LabelEncoder()
model = None

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

# Uses input data to create a model and tests accuracy of the model
def train_model(df, features, target):
    try:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Model Trained", f"Model trained successfully! Accuracy: {accuracy:.2f}")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")
    return None

# Uses model to predict values of the target variable for a given dataset of features
def make_predictions(model, df, features):
    try:
        X_new = df[features]
        predictions = model.predict(X_new)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Predictions:\n{predictions}")
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

