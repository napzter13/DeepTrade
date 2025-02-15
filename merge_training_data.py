import pandas as pd

# List of CSV files (replace with actual file paths)
csv_files = [
    "training_data/training_data.csv",
    "training_data/training_data_2025_2_14.csv",
]

# Read and merge all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)  # ignore_index resets index

# Save merged DataFrame to a new CSV file
merged_df.to_csv("training_data/training_data.csv", index=False)

print("training_data/training_data files merged successfully!")



# List of CSV files (replace with actual file paths)
csv_files = [
    "training_data/rl_transitions.csv",
    "training_data/rl_transitions_2025_2_14.csv",
]

# Read and merge all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)  # ignore_index resets index

# Save merged DataFrame to a new CSV file
merged_df.to_csv("training_data/rl_transitions.csv", index=False)

print("training_data/rl_transitions files merged successfully!")
