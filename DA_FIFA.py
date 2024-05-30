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

#Removing Description Column 
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

# Drop the "Description" column from the list of numeric columns
numeric_cols = [col for col in numeric_cols if col != "Description"]

# print(df.head())

# Drop unneccessary attributes

df_data = df_data.drop('Name.1', axis = 1)
df_data = df_data.drop('Yel', axis = 1)
df_data = df_data.drop('Red', axis = 1)
df_data = df_data.drop('Interceptions per game', axis = 1)
df_data = df_data.drop('Fouls', axis = 1)
df_data = df_data.drop('Offside won per game', axis = 1)
df_data = df_data.drop('Outfielder Block Per Game', axis = 1)
df_data = df_data.drop('OwnG', axis = 1)
df_data = df_data.drop('Offsides per game', axis = 1)
df_data = df_data.drop('Dispossessed per game', axis = 1)
df_data = df_data.drop('Bad control per game', axis = 1)
df_data = df_data.drop('Long balls per game', axis = 1)
df_data = df_data.drop('Through balls per game', axis = 1)
df_data = df_data.drop('Aerials Won per game', axis = 1)
df_data = df_data.drop('Tackles', axis = 1)
df_data = df_data.drop('Clearances per game', axis = 1)
df_data = df_data.drop('Dribbled past per game', axis = 1)
df_data = df_data.drop('Dribbles per game', axis = 1)
df_data = df_data.drop('Fouled per game', axis = 1)
df_data = df_data.drop('Key passes per game', axis = 1)
df_data = df_data.drop('Passes per game', axis = 1)
df_data = df_data.drop('Crosses', axis = 1)

# df_DEF = df_DEF.drop('Name.1', axis = 1)
df_DEF = df_DEF.drop('Yel', axis = 1)
df_DEF = df_DEF.drop('Red', axis = 1)
df_DEF = df_DEF.drop('Interceptions per game', axis = 1)
df_DEF = df_DEF.drop('Fouls', axis = 1)
df_DEF = df_DEF.drop('Offside won per game', axis = 1)
df_DEF = df_DEF.drop('Outfielder Block Per Game', axis = 1)
df_DEF = df_DEF.drop('OwnG', axis = 1)
df_DEF = df_DEF.drop('Offsides per game', axis = 1)
df_DEF = df_DEF.drop('Dispossessed per game', axis = 1)
df_DEF = df_DEF.drop('Bad control per game', axis = 1)
df_DEF = df_DEF.drop('Long balls per game', axis = 1)
df_DEF = df_DEF.drop('Through balls per game', axis = 1)
df_DEF = df_DEF.drop('Dribbled past per game', axis = 1)
df_DEF = df_DEF.drop('Goals', axis = 1)
df_DEF = df_DEF.drop('Shots per \ngame', axis = 1)
df_DEF = df_DEF.drop('Dribbles per game', axis = 1)
df_DEF = df_DEF.drop('Fouled per game', axis = 1)
df_DEF = df_DEF.drop('Key passes per game', axis = 1)
df_DEF = df_DEF.drop('Crosses', axis = 1)
df_DEF = df_DEF.drop('Assists', axis = 1)

# df_MID = df_MID.drop('Name.1', axis = 1)
df_MID = df_MID.drop('Yel', axis = 1)
df_MID = df_MID.drop('Red', axis = 1)
df_MID = df_MID.drop('Interceptions per game', axis = 1)
df_MID = df_MID.drop('Fouls', axis = 1)
df_MID = df_MID.drop('Offside won per game', axis = 1)
df_MID = df_MID.drop('Outfielder Block Per Game', axis = 1)
df_MID = df_MID.drop('OwnG', axis = 1)
df_MID = df_MID.drop('Offsides per game', axis = 1)
df_MID = df_MID.drop('Dispossessed per game', axis = 1)
df_MID = df_MID.drop('Bad control per game', axis = 1)
df_MID = df_MID.drop('Long balls per game', axis = 1)
df_MID = df_MID.drop('Through balls per game', axis = 1)
df_MID = df_MID.drop('Fouled per game', axis = 1)

# df_OFF = df_OFF.drop('Name.1', axis = 1)
df_OFF = df_OFF.drop('Yel', axis = 1)
df_OFF = df_OFF.drop('Red', axis = 1)
df_OFF = df_OFF.drop('Interceptions per game', axis = 1)
df_OFF = df_OFF.drop('Fouls', axis = 1)
df_OFF = df_OFF.drop('Offside won per game', axis = 1)
df_OFF = df_OFF.drop('Outfielder Block Per Game', axis = 1)
df_OFF = df_OFF.drop('OwnG', axis = 1)
df_OFF = df_OFF.drop('Offsides per game', axis = 1)
df_OFF = df_OFF.drop('Dispossessed per game', axis = 1)
df_OFF = df_OFF.drop('Bad control per game', axis = 1)
df_OFF = df_OFF.drop('Long balls per game', axis = 1)
df_OFF = df_OFF.drop('Through balls per game', axis = 1)
df_OFF = df_OFF.drop('Tackles', axis = 1)
df_OFF = df_OFF.drop('Clearances per game', axis = 1)
df_OFF = df_OFF.drop('Dribbled past per game', axis = 1)
df_OFF = df_OFF.drop('Dribbles per game', axis = 1)
df_OFF = df_OFF.drop('Fouled per game', axis = 1)
df_OFF = df_OFF.drop('Key passes per game', axis = 1)
df_OFF = df_OFF.drop('Passes per game', axis = 1)
df_OFF = df_OFF.drop('Pass success percentage', axis = 1)
df_OFF = df_OFF.drop('Crosses', axis = 1)


print("\n General")
print(df.columns.tolist())

print("\n Data")
print(df_data.columns.tolist())

print("\n DEF")
print(df_DEF.columns.tolist())

print("\n MID")
print(df_MID.columns.tolist())

print("\n OFF")
print(df_OFF.columns.tolist())


