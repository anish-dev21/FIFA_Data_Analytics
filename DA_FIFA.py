import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the sheet names containing the data
sheet_names = ["Data", "DEF", "MID", "OFF"]

# Load and concatenate data from all tables
dfs = []
for sheet_name in sheet_names:
    df = pd.read_excel("Data.xlsx", sheet_name=sheet_name)
    df['Table'] = sheet_name  # Add a column to identify the source table
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Remove unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Rename the column "Name.1" to "Description"
data.rename(columns={"Name.1": "Description"}, inplace=True)

#Removing Description Column 
data = data.drop(columns=["Description"])

# Check for missing values
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# Fill missing values for numeric columns with mean
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Check if missing values have been filled
print("\nMissing Values After Filling:")
print(data.isnull().sum())

# Drop the "Description" column from the list of numeric columns
numeric_cols = [col for col in numeric_cols if col != "Description"]

# Summary statistics of numerical features (excluding the "Description" column)
print("\nSummary Statistics (Using Only Numeric Values, Excluding 'Description' column):")
print(data[numeric_cols].describe())

# Visualize missing values using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cmap='viridis', yticklabels=False, cbar=False)
plt.title('Missing Values Heatmap')
plt.show()





