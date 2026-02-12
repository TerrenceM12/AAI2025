import pandas as pd
import numpy as np

# 1. Setup
num_records = 1000
np.random.seed(42)  # Ensures you get the same "random" numbers every time

# 2. Generate Feature Columns
# Age: Random integer between 18 and 70
age = np.random.randint(18, 70, num_records)

# Monthly Usage: Random integer between 5 and 100 hours
monthly_usage_hours = np.random.randint(5, 100, num_records)

# Purchase Amount: Random float between 20.00 and 500.00
purchase_amount = np.round(np.random.uniform(20, 500, num_records), 2)

# Customer Service Calls: Random integer between 0 and 10
# We weight it slightly so lower numbers are more common (0-4), but spikes happen
customer_service_calls = np.random.randint(0, 10, num_records)

# Region: Random choice
region = np.random.choice(['North', 'South', 'East', 'West'], num_records)

# 3. Generate Target (Churn) with Logic
# We create a "Churn Probability Score" based on the other columns.
# Logic: 
#   - More service calls (+ effect) -> Higher churn
#   - Higher usage (- effect) -> Lower churn (loyal customer)
#   - Higher purchase (- effect) -> Lower churn
score = (customer_service_calls * 15) - (monthly_usage_hours * 0.5) - (purchase_amount * 0.1)

# Add random noise so the model isn't 100% perfect (simulating real life)
score += np.random.normal(0, 20, num_records)

# Convert score to 0 or 1. If score > threshold, they churn.
# The threshold -20 is chosen to balance the dataset approx 50/50 or 70/30.
churn = np.where(score > -20, 1, 0)

# 4. Create DataFrame
df = pd.DataFrame({
    'age': age,
    'monthly_usage_hours': monthly_usage_hours,
    'purchase_amount': purchase_amount,
    'customer_service_calls': customer_service_calls,
    'region': region,
    'churn': churn
})

# 5. Save to CSV
df.to_csv('customer_churn_large.csv', index=False)

print(f"Dataset with {len(df)} records created.")
print(df.head())
