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

df_data = pd.read_excel("Data.xlsx", sheet_name='Data')
df_DEF = pd.read_excel("Data.xlsx", sheet_name='DEF')
df_MID = pd.read_excel("Data.xlsx", sheet_name='MID')
df_OFF = pd.read_excel("Data.xlsx", sheet_name='OFF')

data = pd.concat(dfs, ignore_index=True)

# Remove unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Rename the column "Name.1" to "Description"
data.rename(columns={"Name.1": "Description"}, inplace=True)

# Remove Description Column
data = data.drop(columns=["Description"])

# Check for missing values
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# Fill missing values for numeric columns with mean
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Fill missing values for categorical columns with mode
categorical_cols = ['Name', 'Rating', 'Fifa Ability Overall', 'Position']
for col in categorical_cols:
    mode_value = data[col].mode()[0]
    data[col] = data[col].fillna(mode_value)

# Drop rows with any remaining missing values
data = data.dropna()

# Check if missing values have been filled
print("\nMissing Values After Filling:")
print(data.isnull().sum())

# Calculate average minutes played per game
df_data['Avg Minutes Played'] = np.where(df_data['Apps'] > 0, df_data['Minutes played'] / df_data['Apps'], np.nan)
df_DEF['Avg Minutes Played'] = np.where(df_DEF['Apps'] > 0, df_DEF['Minutes played'] / df_DEF['Apps'], np.nan)
df_MID['Avg Minutes Played'] = np.where(df_MID['Apps'] > 0, df_MID['Minutes played'] / df_MID['Apps'], np.nan)
df_OFF['Avg Minutes Played'] = np.where(df_OFF['Apps'] > 0, df_OFF['Minutes played'] / df_OFF['Apps'], np.nan)


# Drop unnecessary attributes in individual DataFrames
df_data = df_data.drop(columns=[
    'Name.1', 'Yel', 'Red', 'Interceptions per game', 'Fouls',
    'Offside won per game', 'Outfielder Block Per Game', 'OwnG', 'Offsides per game',
    'Dispossessed per game', 'Bad control per game', 'Long balls per game', 
    'Through balls per game', 'Aerials Won per game', 'Tackles', 'Clearances per game', 
    'Dribbled past per game', 'Dribbles per game', 'Fouled per game', 'Key passes per game', 
    'Passes per game', 'Crosses', 'Minutes played', 'Shots pergame', 'Assists', 'Goals'
])

df_DEF = df_DEF.drop(columns=[
    'Yel', 'Red', 'Interceptions per game', 'Fouls', 'Offside won per game', 
    'Outfielder Block Per Game', 'OwnG', 'Offsides per game', 'Dispossessed per game', 
    'Bad control per game', 'Long balls per game', 'Through balls per game', 
    'Dribbled past per game', 'Goals', 'Shots per \ngame', 'Dribbles per game', 
    'Fouled per game', 'Key passes per game', 'Crosses', 'Assists', 'Minutes played'
])

df_MID = df_MID.drop(columns=[
    'Yel', 'Red', 'Interceptions per game', 'Fouls', 'Offside won per game', 
    'Outfielder Block Per Game', 'OwnG', 'Offsides per game', 'Dispossessed per game', 
    'Bad control per game', 'Long balls per game', 'Through balls per game', 
    'Fouled per game', 'Minutes played'
])

df_OFF = df_OFF.drop(columns=[
    'Yel', 'Red', 'Interceptions per game', 'Fouls', 'Offside won per game', 
    'Outfielder Block Per Game', 'OwnG', 'Offsides per game', 'Dispossessed per game', 
    'Bad control per game', 'Long balls per game', 'Through balls per game', 
    'Tackles', 'Clearances per game', 'Dribbled past per game', 'Dribbles per game', 
    'Fouled per game', 'Key passes per game', 'Passes per game', 'Pass success percentage', 
    'Crosses', 'Minutes played'
])

columns_to_drop = []
df_data = df_data.drop(columns=columns_to_drop)

print("\n General")
print(data.columns.tolist())

print("\n Data")
print(df_data.columns.tolist())

print("\n DEF")
print(df_DEF.columns.tolist())

print("\n MID")
print(df_MID.columns.tolist())

print("\n OFF")
print(df_OFF.columns.tolist())

# Export the consolidated DataFrame
data.to_csv('modified_data.csv', index=False)

# Optionally, export individual DataFrames
df_data.to_csv('modified_df_data.csv', index=False)
df_DEF.to_csv('modified_df_DEF.csv', index=False)
df_MID.to_csv('modified_df_MID.csv', index=False)
df_OFF.to_csv('modified_df_OFF.csv', index=False)  