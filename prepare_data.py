import pandas as pd
import os

# Read the separate files
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Add label column
fake_df['label'] = 0  # 0 = fake
true_df['label'] = 1  # 1 = real

# Combine them
combined_df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the data
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check what columns we have
print("Columns in dataset:", combined_df.columns.tolist())
print("Total rows:", len(combined_df))
print("\nFirst few rows:")
print(combined_df.head())

# Save as train.csv
combined_df.to_csv('data/train.csv', index=False)
print("\n✅ Created data/train.csv successfully!")

 