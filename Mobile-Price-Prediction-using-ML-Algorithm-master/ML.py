import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("mobile_data.csv")

# Map the 'price_range' column to combine "Very High" (3) into "High" (2)
data['price_range'] = data['price_range'].replace({3: 2})  # Combine 3 into 2

# Separate features and target variable
X = data.drop(columns=["price_range"])  # Features
y = data["price_range"]  # Target (modified price range)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the model for prediction
joblib.dump(model, "model.pkl")
