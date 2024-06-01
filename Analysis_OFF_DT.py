##### scatter plot
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# try:
#     # Step 1: Load the data
#     file_path = 'modified_df_OFF.csv'
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

#     # Step 5: Train a Decision Tree regressor model
#     regressor = DecisionTreeRegressor(random_state=42)
#     regressor.fit(X_train, y_train)

#     # Step 6: Make predictions on the test set
#     y_pred = regressor.predict(X_test)

#     # Step 7: Evaluate the model
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print("Mean Squared Error:", mse)
#     print("R^2 Score:", r2)

#     # Step 8: Calculate the number of matching values and accuracy percentage
#     tolerance = 1.5  # Define a tolerance level for matching values
#     matching_values = np.sum(np.abs(y_test - y_pred) <= tolerance)
#     total_values = len(y_test)
#     accuracy_percentage = (matching_values / total_values) * 100

#     print("Number of matching values:", matching_values)
#     print("Total values:", total_values)
#     print("Percentage of accuracy:", accuracy_percentage)

#     # Step 9: Plotting the results with different colors for points based on prediction error
#     error = y_test - y_pred
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_test, y_pred, c=error, cmap='coolwarm', alpha=0.7)
#     plt.colorbar(label='Prediction Error')
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Plotting the diagonal line
#     plt.xlabel('Actual Ability Overall')
#     plt.ylabel('Predicted FIFA Ability Overall')
#     plt.title('Decision Tree Regression: Actual vs Predicted FIFA Ability Overall')

#     # Add annotations for matching values and accuracy percentage
#     plt.annotate(f'Matching Values: {matching_values}', xy=(0.05, 0.95), xycoords='axes fraction', color='blue')
#     plt.annotate(f'Accuracy Percentage: {accuracy_percentage:.2f}%', xy=(0.05, 0.90), xycoords='axes fraction', color='blue')

#     plt.grid(True)
#     plt.show()

# except Exception as e:
#     print("An error occurred:", e)




####### distribution plot
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# try:
#     # Step 1: Load the data
#     file_path = 'modified_df_OFF.csv'
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

#     # Step 5: Train a Decision Tree regressor model
#     regressor = DecisionTreeRegressor(random_state=42)
#     regressor.fit(X_train, y_train)

#     # Step 6: Make predictions on the test set
#     y_pred = regressor.predict(X_test)

#     # Step 7: Evaluate the model
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print("Mean Squared Error:", mse)
#     print("R^2 Score:", r2)

#     # Step 8: Calculate the number of matching values and accuracy percentage
#     tolerance = 1.5  # Define a tolerance level for matching values
#     matching_values = np.sum(np.abs(y_test - y_pred) <= tolerance)
#     total_values = len(y_test)
#     accuracy_percentage = (matching_values / total_values) * 100

#     print("Number of matching values:", matching_values)
#     print("Total values:", total_values)
#     print("Percentage of accuracy:", accuracy_percentage)

#     # Step 9: Plotting the distribution plot of prediction errors
#     error = y_test - y_pred
#     plt.figure(figsize=(10, 6))
#     sns.histplot(error, kde=True, color='skyblue')
#     plt.xlabel('Prediction Error')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Prediction Errors')
#     plt.grid(True)
#     plt.show()

# except Exception as e:
#     print("An error occurred:", e)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Load the data
    file_path = 'modified_df_OFF.csv'
    df = pd.read_csv(file_path)

    # Preprocess the data
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df['Fifa Ability Overall'] = pd.to_numeric(df['Fifa Ability Overall'], errors='coerce')
    df.dropna(subset=['Fifa Ability Overall'], inplace=True)
    df = df.select_dtypes(include=np.number)

    if df.empty:
        raise ValueError("No numeric columns remaining after preprocessing")

    # Define the feature and target variables
    X = df.drop(columns=['Fifa Ability Overall'])
    y = df['Fifa Ability Overall']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Decision Tree regressor model
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = regressor.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    # Calculate the number of matching values and accuracy percentage
    tolerance = 1.5
    matching_values = np.sum(np.abs(y_test - y_pred) <= tolerance)
    total_values = len(y_test)
    accuracy_percentage = (matching_values / total_values) * 100
    print("Number of matching values:", matching_values)
    print("Total values:", total_values)
    print("Percentage of accuracy:", accuracy_percentage)

    # Plotting the joint plot
    plt.figure(figsize=(10, 6))
    sns.jointplot(x=y_test, y=y_pred, kind='reg', color='skyblue', joint_kws={'scatter_kws': {'alpha': 0.7}})
    plt.suptitle('Decision Tree Regression: Actual vs Predicted FIFA Ability Overall', y=1.02)
    plt.xlabel('Actual Ability Overall')
    plt.ylabel('Predicted FIFA Ability Overall')

    # Plot the Decision Tree
    plt.figure(figsize=(20,10))
    plot_tree(regressor, filled=True, feature_names=X.columns)
    plt.title("Decision Tree Visualization")
    plt.show()

except Exception as e:
    print("An error occurred:", e)






    