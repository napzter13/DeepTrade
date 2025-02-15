import pandas as pd
import glob

# List of CSV files (replace with actual file paths)
csv_files = [
    "output/training_data_2020.csv",
    "output/training_data_2021.csv",
    "output/training_data_2022.csv",
    "output/training_data_2023.csv",
    "output/training_data_2024.csv",
]

# Read and merge all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)  # ignore_index resets index

# Save merged DataFrame to a new CSV file
merged_df.to_csv("output/training_data.csv", index=False)

print("output/training_data files merged successfully!")



# List of CSV files (replace with actual file paths)
csv_files = [
    "output/rl_transitions_2020.csv",
    "output/rl_transitions_2021.csv",
    "output/rl_transitions_2022.csv",
    "output/rl_transitions_2023.csv",
    "output/rl_transitions_2024.csv",
]

# Read and merge all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)  # ignore_index resets index

# Save merged DataFrame to a new CSV file
merged_df.to_csv("output/rl_transitions.csv", index=False)

print("output/rl_transitions files merged successfully!")
