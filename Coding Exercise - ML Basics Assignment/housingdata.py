import pandas as pd
import numpy as np

# 1. Setup
num_records = 1000
np.random.seed(42)

# 2. Generate Square Footage
# Random sizes between 1000 and 3500 sq ft
square_footage = np.random.randint(1000, 3500, num_records)

# 3. Generate Location
# Randomly assign 'Downtown', 'Suburb', or 'Rural'
locations = np.random.choice(['Downtown', 'Suburb', 'Rural'], num_records)

# 4. Generate Price (with Location Logic)
# We calculate a 'base price' and then apply a multiplier based on location
# Downtown = expensive (1.5x), Suburb = moderate (1.2x), Rural = cheaper (1.0x)

base_price_per_sqft = 150
noise = np.random.normal(0, 15000, num_records)  # Add randomness so it's not perfect

# Create a temporary multiplier array based on location
location_multipliers = []
for loc in locations:
    if loc == 'Downtown':
        location_multipliers.append(1.5)
    elif loc == 'Suburb':
        location_multipliers.append(1.2)
    else: # Rural
        location_multipliers.append(0.9)

# Calculate final price
price = (square_footage * base_price_per_sqft * np.array(location_multipliers)) + noise
price = np.round(price, -2) # Round to nearest 100

# 5. Create DataFrame with your specific columns
df = pd.DataFrame({
    'square_footage': square_footage,
    'location': locations,
    'price': price
})

# 6. Save to CSV (Run this to get your file!)
df.to_csv('housing_data_v2.csv', index=False)

print(f"Dataset created with {len(df)} records.")
print(df.head(10))
