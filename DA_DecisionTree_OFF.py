
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
    X = df.drop(columns=['Fifa Ability Overall']) #feature
    y = df['Fifa Ability Overall'] #target

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

    # Plot the Decision Tree
    plt.figure(figsize=(20,10))
    plot_tree(regressor, filled=True, feature_names=X.columns)
    plt.title("Decision Tree Visualization")
    plt.show()

except Exception as e:
    print("An error occurred:", e)