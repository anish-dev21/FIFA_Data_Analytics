# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# try:
#     # Step 1: Load the data
#     file_path = 'modified_df_DEF.csv'
#     df = pd.read_csv(file_path)

#     # Step 2: Preprocess the data
#     # Drop any unnamed columns
#     df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

#     # Fill missing values - Filling numeric columns with mean
#     numeric_cols = df.select_dtypes(include=np.number).columns
#     df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

#     # Ensure 'Fifa Ability Overall' is numeric
#     df['Fifa Ability Overall'] = pd.to_numeric(df['Fifa Ability Overall'], errors='coerce')

#     # Drop rows with missing target values
#     df.dropna(subset=['Fifa Ability Overall'], inplace=True)

#     # Drop non-numeric columns that cannot be used in regression
#     df = df.select_dtypes(include=np.number)

#     if df.empty:
#         raise ValueError("No numeric columns remaining after preprocessing")

#     # Print column names after preprocessing
#     print("Column names after preprocessing:\n", df.columns)

#     # Step 3: Define the feature and target variables
#     # Features: All other characteristics except FIFA Ability Overall
#     X = df.drop(columns=['Fifa Ability Overall'])

#     # Target: FIFA Ability Overall
#     y = df['Fifa Ability Overall']

#     # Step 4: Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Step 5: Train a linear regression model
#     regressor = LinearRegression()
#     regressor.fit(X_train, y_train)

#     # Step 6: Make predictions on the test set
#     y_pred = regressor.predict(X_test)

#     # Step 7: Evaluate the model
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print("Mean Squared Error:", mse)
#     print("R^2 Score:", r2)

#     # Step 8: Plotting the results with different colors for points based on prediction error
#     error = y_test - y_pred
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_test, y_pred, c=error, cmap='coolwarm', alpha=0.7)
#     plt.colorbar(label='Prediction Error')
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Plotting the diagonal line
#     plt.xlabel('Actual Ability Overall')
#     plt.ylabel('Predicted FIFA Ability Overall')
#     plt.title('Linear Regression: Actual vs Predicted FIFA Ability Overall')
#     plt.grid(True)
#     plt.show()

# except Exception as e:
#     print("An error occurred:", e)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the data
file_path = 'modified_df_DEF.csv'
df = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Drop any unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Fill missing values - Filling numeric columns with mean
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Ensure 'Fifa Ability Overall' is numeric
df['Fifa Ability Overall'] = pd.to_numeric(df['Fifa Ability Overall'], errors='coerce')

# Drop rows with missing target values
df.dropna(subset=['Fifa Ability Overall'], inplace=True)

# Print the column names to verify
print("Column names in the DataFrame:\n", df.columns)

# Step 3: Define the feature and target variables
# Adjust the feature list based on the actual column names
features = ['Rating', 'Apps', 'Minutes played', 'Assists', 'Man of the match', 'Goals', 'Shots per game', 'Pass success percentage']

# Check for missing columns in the DataFrame
missing_features = [feature for feature in features if feature not in df.columns]
if missing_features:
    print("The following features are missing from the DataFrame:", missing_features)
    # Adjust the features list by removing missing columns
    features = [feature for feature in features if feature in df.columns]

X = df[features]

# Target: FIFA Ability Overall
y = df['Fifa Ability Overall']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = regressor.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Step 8: Visualize the results using a joint plot
plt.figure(figsize=(10, 6))
sns.jointplot(x=y_test, y=y_pred, kind='reg', height=8)
plt.xlabel('Actual FIFA Ability Overall')
plt.ylabel('Predicted FIFA Ability Overall')
plt.suptitle('Decision Tree Regression: Actual vs Predicted FIFA Ability Overall', y=1.02)
plt.show()