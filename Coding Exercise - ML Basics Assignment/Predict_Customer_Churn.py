import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load the Data
try:
    # Fixed typo in filename: 'chun' -> 'churn'
    df = pd.read_csv('customer_churn_large.csv')
except FileNotFoundError:
    print("File not found! Please generate the CSV first.")
    df = pd.DataFrame()

# 2. Define Features and Target
X = df[['age', 'monthly_usage_hours', 'purchase_amount', 'customer_service_calls', 'region']]
y = df['churn']

# 3. Preprocessing
# Note: 'num' comes FIRST, 'cat' comes SECOND. This order matters for the labels later.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'monthly_usage_hours', 'purchase_amount', 'customer_service_calls']),
        ('cat', OneHotEncoder(sparse_output=False), ['region'])
    ]
)

# 4. Create Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model
model.fit(X_train, y_train)

# 7. Predict for a New Customer
new_customer = pd.DataFrame({
    'age': [35],
    'monthly_usage_hours': [20],
    'purchase_amount': [150],
    'customer_service_calls': [5],
    'region': ['West']
})

# Get probability of class 1 (Churn)
churn_probability = model.predict_proba(new_customer)[0][1]

# Classify based on threshold
threshold = 0.5
churn_prediction = 1 if churn_probability > threshold else 0

print(f"Churn Probability: {churn_probability:.2f}")
print(f"Prediction: {'Churn' if churn_prediction == 1 else 'No Churn'}")

# 8. Display Model Coefficients
# FIX: We must match the order of the ColumnTransformer (Num first, then Cat)
num_features = ['age', 'monthly_usage_hours', 'purchase_amount', 'customer_service_calls']
cat_features = (model.named_steps['preprocessor']
                .named_transformers_['cat']
                .get_feature_names_out(['region'])).tolist()

feature_names = num_features + cat_features 
coefficients = model.named_steps['classifier'].coef_[0]

print("\nModel Coefficients:")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.2f}")
