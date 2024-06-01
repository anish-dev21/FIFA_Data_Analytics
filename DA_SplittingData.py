import pandas as pd
from sklearn.model_selection import train_test_split

# List of CSV files
csv_files = ['modified_df_data.csv', 'modified_df_DEF.csv', 'modified_df_MID.csv', 'modified_df_OFF.csv']

# Dictionary to store training and testing data
data_dict = {}

# Loop through each CSV file
for csv_file in csv_files:
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Features and target
    X = df.drop(columns=['Fifa Ability Overall'])  # Features are all columns except 'Fifa ability overall'
    y = df['Fifa Ability Overall']  # Target column
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Store the data in the dictionary
    data_dict[csv_file] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
#End of the actual code that splits the Data 


#Putting in extra prints to check if the data was split correctly 
    # Load the data from CSV files
df_data = pd.read_csv("modified_df_data.csv")
df_DEF = pd.read_csv("modified_df_DEF.csv")
df_MID = pd.read_csv("modified_df_MID.csv")
df_OFF = pd.read_csv("modified_df_OFF.csv")

# List of CSV files and their corresponding dataframes
csv_files = [("modified_df_data.csv", df_data),
             ("modified_df_DEF.csv", df_DEF),
             ("modified_df_MID.csv", df_MID),
             ("modified_df_OFF.csv", df_OFF)]

# Loop through each CSV file and perform train-test split
for csv_file, df in csv_files:
    # Extract features and target
    X = df.drop("Fifa Ability Overall", axis=1)
    y = df["Fifa Ability Overall"]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print information about the split
    print(f"CSV File: {csv_file}")
    print("Training Set Shape:", X_train.shape)
    print("Testing Set Shape:", X_test.shape)
    print("Target Distribution in Training Set:")
    print(y_train.value_counts())
    print("Target Distribution in Testing Set:")
    print(y_test.value_counts())
    print("="*50)
