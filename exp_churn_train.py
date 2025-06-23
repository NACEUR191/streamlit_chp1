import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'Expresso_churn_dataset.csv')

from sklearn.preprocessing import LabelEncoder
# Encoding categorical variables
label_enc = LabelEncoder()
df['REGION'] = label_enc.fit_transform(df['REGION'])
df['TOP_PACK'] = label_enc.fit_transform(df['TOP_PACK'])

df['TENURE'] = df['TENURE'] == "K > 24 month" # Convert to boolean
df = df.drop(['user_id'], axis=1)
df = df.drop(['MRG'], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# Prepare data for modeling
# Ensure 'CHURN' is the target variable
X = df.drop('CHURN', axis=1)  # Ensure 'churn' is the label
y = df['CHURN']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train a Random Forest model 
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

import joblib
joblib.dump(model, "churn_model.pkl")# Save the model
print("Model trained and saved as 'churn_model.pkl'")