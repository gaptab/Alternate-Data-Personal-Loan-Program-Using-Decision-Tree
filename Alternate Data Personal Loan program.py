import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# Step 1: Generate Dummy Data
np.random.seed(42)

num_samples = 5000
data = pd.DataFrame({
    'credit_score': np.random.randint(300, 850, num_samples),
    'income': np.random.randint(20000, 150000, num_samples),
    'loan_amount': np.random.randint(5000, 50000, num_samples),
    'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], num_samples, p=[0.7, 0.2, 0.1]),
    'loan_purpose': np.random.choice(['Education', 'Medical', 'Business', 'Home Renovation'], num_samples),
    'previous_defaults': np.random.randint(0, 5, num_samples),
    'risk_label': np.random.choice([0, 1], num_samples, p=[0.91, 0.09])  # Initial risk: 9% defaults
})

# Step 2: Preprocessing
data['employment_status'] = data['employment_status'].map({'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2})
data['loan_purpose'] = data['loan_purpose'].astype('category').cat.codes

# Features & Target
X = data.drop(columns=['risk_label'])
y = data['risk_label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Decision Tree Model
model = DecisionTreeClassifier(max_depth=5, min_samples_split=50, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Step 4: Extract Business Rules from Decision Tree
tree_rules = export_text(model, feature_names=X.columns.tolist())
print("Extracted Business Rules:\n", tree_rules)

# Step 5: Measure Risk Reduction
initial_risk = (y_test.sum() / len(y_test)) * 100  # Before applying decision tree rules
filtered_data = X_test[model.predict(X_test) == 0]  # Customers predicted as low risk
new_risk = (y_test.loc[filtered_data.index].sum() / len(filtered_data)) * 100  # After applying rules

print(f"\nRisk Reduction: {initial_risk:.2f}% → {new_risk:.2f}%")

# Save the dataset to a CSV file
data.to_csv("alternate_data_personal_loan.csv", index=False)
print("Dataset saved as 'alternate_data_personal_loan.csv'")
