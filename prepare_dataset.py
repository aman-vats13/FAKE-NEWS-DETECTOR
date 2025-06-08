import pandas as pd

# Load both CSVs
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels
fake_df['label'] = 0
true_df['label'] = 1

# Combine them
combined_df = pd.concat([fake_df, true_df], ignore_index=True)

# Optional: drop unnecessary columns
combined_df = combined_df[['title', 'text', 'label']]

# Save as train.csv
combined_df.to_csv("dataset/train.csv", index=False)

print("✅ Dataset prepared as dataset/train.csv")
