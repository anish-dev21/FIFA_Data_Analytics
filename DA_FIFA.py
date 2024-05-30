import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "Data.xlsx"
data = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
print("Dataset Overview:")
print(data.head())

# Check the dimensions of the dataset
print("\nDataset Dimensions:")
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])

# Summary statistics of numerical features
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# Fill missing values
# Example: For numerical columns, fill missing values with mean
numerical_cols = data.select_dtypes(include='number').columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

# For categorical columns, fill missing values with mode
categorical_cols = data.select_dtypes(include='object').columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Check if missing values have been filled
print("\nMissing Values After Filling:")
print(data.isnull().sum())

# Visualize missing values using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cmap='viridis', yticklabels=False, cbar=False)
plt.title('Missing Values Heatmap')
plt.show()

# Now, you can proceed with further analysis, such as visualization or model building

