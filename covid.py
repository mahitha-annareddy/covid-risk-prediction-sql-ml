# ==============================
# 1. Import Libraries
# ==============================
import numpy as np
import pandas as pd
import sqlite3

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================
# 2. Store Data in SQL Database
# ==============================
# Connect to SQLite database
conn = sqlite3.connect("covid_patients.db")

# Read CSV file
df = pd.read_csv("disease_risk_dataset.csv")

# Store data into SQL table
df.to_sql("patients", conn, if_exists="replace", index=False)

print("✅ Data stored successfully in SQL database")


# ==============================
# 3. Fetch Data from SQL
# ==============================
query = "SELECT * FROM patients"
dataset = pd.read_sql(query, conn)

print("\n📊 Dataset from SQL:")
print(dataset.head())


# ==============================
# 4. Split Independent & Dependent Variables
# ==============================
X = dataset.iloc[:, :-1].values   # Features
y = dataset.iloc[:, -1].values    # Target


# ==============================
# 5. Handle Missing Values
# ==============================
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


# ==============================
# 6. Encode Target Variable
# ==============================
le = LabelEncoder()
y = le.fit_transform(y)   # Yes = 1, No = 0


# ==============================
# 7. Train–Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 8. Feature Scaling
# ==============================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ==============================
# 9. Train ML Model
# ==============================
model = LogisticRegression()
model.fit(X_train, y_train)


# ==============================
# 10. Prediction
# ==============================
y_pred = model.predict(X_test)


# ==============================
# 11. Model Evaluation
# ==============================
print("\n✅ Model Evaluation Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# ==============================
# 12. Close Database Connection
# ==============================
conn.close()
