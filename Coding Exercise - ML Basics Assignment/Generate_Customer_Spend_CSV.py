import pandas as pd
import numpy as np

# 1. Setup
num_records = 1000
np.random.seed(42)  # Ensures reproducible results

# 2. Generate Independent Features
# Age: Random integer between 18 and 75
age = np.random.randint(18, 75, num_records)

# Region: Randomly chosen from the list
regions = np.random.choice(['North', 'South', 'East', 'West'], num_records)

# Purchase Frequency: Random integer between 1 and 50 times a year
purchase_frequency = np.random.randint(1, 50, num_records)

# 3. Generate Dependent Feature (Spending)
# Logic: Spending is roughly (Frequency * Avg Transaction Value)
# Let's assume an average transaction is around $100, but varies by person ($50-$150)
avg_transaction_value = np.random.uniform(50, 150, num_records) 

# Calculate Annual Spending based on frequency
annual_spending = purchase_frequency * avg_transaction_value

# Add some "noise" (random variation) so it's not a perfect straight line
noise = np.random.normal(0, 200, num_records)
annual_spending += noise

# Clean up: Round to 2 decimal places and ensure no negative spending
annual_spending = np.round(np.abs(annual_spending), 2)

# 4. Create DataFrame
df = pd.DataFrame({
    'annual_spending': annual_spending,
    'purchase_frequency': purchase_frequency,
    'age': age,
    'region': regions
})

# 5. Save to CSV
csv_filename = 'customer_spending.csv'
df.to_csv(csv_filename, index=False)

print(f"File '{csv_filename}' created with {len(df)} records.")
print(df.head())
