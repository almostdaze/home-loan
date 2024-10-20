import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('loan.csv')

# Features and target variable
X = df[['credit_score', 'income', 'debt_to_income_ratio', 'down_payment']]
y = df['loan_approved']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train a Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Evaluate both models
rf_predictions = rf_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))

# Choose the Random Forest model and save it
with open('home_loan_prediction_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Uncomment the following lines if you prefer to save the Logistic Regression model instead
# with open('home_loan_prediction_model.pkl', 'wb') as f:
#     pickle.dump(lr_model, f)